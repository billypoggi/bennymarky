--  Ensure the table exists
CREATE TABLE IF NOT EXISTS performance_test (
    id SERIAL PRIMARY KEY,
    num INT,
    data TEXT
);
