-- users
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  name VARCHAR(255),
  points INT DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- categories (коэффициенты расчёта)
CREATE TABLE categories (
  id SERIAL PRIMARY KEY,
  key VARCHAR(100) UNIQUE NOT NULL, -- e.g. "food", "transport_bus", "transport_car", "groceries"
  title VARCHAR(255),
  factor NUMERIC NOT NULL, -- коэффициент (kgCO2e per unit), единицы оговариваются в API (например per RUB or per km)
  unit VARCHAR(50) NOT NULL -- e.g. "per_rub", "per_km", "per_item"
);

-- transactions (покупки/поездки) — записи для расчёта следа
CREATE TABLE transactions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  category_id INT REFERENCES categories(id),
  amount NUMERIC NOT NULL, -- денежная сумма или расстояние, в зависимости от unit
  currency VARCHAR(10) DEFAULT 'RUB',
  footprint_kg NUMERIC NOT NULL, -- рассчитанный kg CO2e
  metadata JSONB, -- дополнительная информация (merchant, city)
  created_at TIMESTAMPTZ DEFAULT now()
);

-- challenges (челленджи / задания)
CREATE TABLE challenges (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title VARCHAR(255) NOT NULL,
  description TEXT,
  points_reward INT DEFAULT 0,
  starts_at TIMESTAMPTZ,
  ends_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- user_challenges (участие пользователя в челлендже)
CREATE TABLE user_challenges (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  challenge_id UUID REFERENCES challenges(id) ON DELETE CASCADE,
  status VARCHAR(20) DEFAULT 'joined', -- joined/completed
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- recycling_points (точки сбора)
CREATE TABLE recycling_points (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title VARCHAR(255),
  description TEXT,
  lat NUMERIC,
  lon NUMERIC,
  tags TEXT[], -- e.g. ['plastic','glass']
  created_at TIMESTAMPTZ DEFAULT now()
);
