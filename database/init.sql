CREATE DATABASE IF NOT EXISTS sign_language_db;
USE sign_language_db;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sign_gestures (
    id INT AUTO_INCREMENT PRIMARY KEY,
    gesture_name VARCHAR(50) NOT NULL UNIQUE,
    description TEXT,
    category VARCHAR(50) DEFAULT 'ASL',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS recognition_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_end TIMESTAMP NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS translation_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT,
    gesture_id INT,
    confidence FLOAT NOT NULL,
    translated_text VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES recognition_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (gesture_id) REFERENCES sign_gestures(id) ON DELETE SET NULL
);

INSERT INTO sign_gestures (gesture_name, description) VALUES
('HELLO', 'Greeting sign hello'),
('THANK_YOU', 'Gratitude sign thank you'),
('PLEASE', 'Polite request sign please'),
('YES', 'Affirmative sign yes'),
('NO', 'Negative sign no'),
('HELP', 'Assistance sign help'),
('I_LOVE_YOU', 'Affection sign I love you'),
('GOOD', 'Positive evaluation sign good'),
('STOP', 'Halt sign stop');
