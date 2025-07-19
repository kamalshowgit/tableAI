-- Create table
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    salary REAL,
    join_date TEXT,
    is_active INTEGER,
    department TEXT
);

-- Insert sample data
INSERT INTO employees (name, age, salary, join_date, is_active, department) VALUES
('Alice', 30, 70000, '2020-01-15', 1, 'Engineering'),
('Bob', 28, 65000, '2021-03-22', 1, 'Marketing'),
('Charlie', 35, 80000, '2019-07-10', 0, 'Sales');