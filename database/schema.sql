-- =====================================================================
-- Phase 15: Enterprise Banking Loan Risk Management Database Schema
-- Compatible with MySQL 8.0+ / MariaDB
-- =====================================================================

CREATE DATABASE IF NOT EXISTS loan_risk_db
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE loan_risk_db;

-- 1. Table: Customers (Demographic and identity details)
CREATE TABLE IF NOT EXISTS customers (
    customer_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_uuid VARCHAR(36) NOT NULL UNIQUE,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    age INT NOT NULL,
    marital_status ENUM('single', 'married') NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    profession VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_customer_state (state),
    INDEX idx_customer_profession (profession)
) ENGINE=InnoDB;

-- 2. Table: Employment & Assets (Financial stability indicators)
CREATE TABLE IF NOT EXISTS customer_financials (
    financial_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT NOT NULL,
    annual_income DECIMAL(15, 2) NOT NULL,
    work_experience_years INT NOT NULL,
    current_job_years INT NOT NULL,
    house_ownership ENUM('owned', 'rented', 'norent_noown') NOT NULL,
    current_house_years INT NOT NULL,
    car_ownership ENUM('yes', 'no') NOT NULL,
    verified_at TIMESTAMP NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE,
    INDEX idx_customer_financials (customer_id)
) ENGINE=InnoDB;

-- 3. Table: Loan Applications
CREATE TABLE IF NOT EXISTS loan_applications (
    application_id INT AUTO_INCREMENT PRIMARY KEY,
    application_number VARCHAR(50) NOT NULL UNIQUE,
    customer_id INT NOT NULL,
    requested_amount DECIMAL(15, 2) NOT NULL,
    loan_purpose VARCHAR(100) DEFAULT 'Personal/General',
    submission_channel ENUM('web_portal', 'api', 'branch') DEFAULT 'web_portal',
    application_status ENUM('pending', 'under_review', 'approved', 'rejected', 'escalated') DEFAULT 'pending',
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    decided_at TIMESTAMP NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE,
    INDEX idx_application_status (application_status)
) ENGINE=InnoDB;

-- 4. Table: Prediction History (AI Model Audit Log)
CREATE TABLE IF NOT EXISTS prediction_history (
    prediction_id INT AUTO_INCREMENT PRIMARY KEY,
    application_id INT NOT NULL,
    model_version VARCHAR(50) NOT NULL DEFAULT 'ANN_Model_C_v1.0',
    model_architecture VARCHAR(100) DEFAULT 'Input(11)-Dense(32)-Dense(16)-Dense(8)-Dense(1)',
    raw_sigmoid_output DECIMAL(6, 4) NOT NULL,
    risk_percentage DECIMAL(5, 2) NOT NULL,
    predicted_risk ENUM('Low Risk', 'High Risk') NOT NULL,
    confidence_score DECIMAL(5, 2) NOT NULL,
    risk_tier ENUM('Ultra Low Risk', 'Low Risk', 'Moderate Risk', 'High Risk', 'Critical Risk') NOT NULL,
    automated_recommendation TEXT,
    features_payload JSON NOT NULL,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (application_id) REFERENCES loan_applications(application_id) ON DELETE CASCADE,
    INDEX idx_predicted_risk (predicted_risk),
    INDEX idx_risk_tier (risk_tier)
) ENGINE=InnoDB;

-- 5. Table: Underwriter Decisions (Human-in-the-loop audit trail)
CREATE TABLE IF NOT EXISTS underwriting_decisions (
    decision_id INT AUTO_INCREMENT PRIMARY KEY,
    application_id INT NOT NULL,
    officer_id VARCHAR(50) NOT NULL,
    officer_name VARCHAR(100) NOT NULL,
    decision ENUM('approved', 'declined', 'counter_offer', 'requested_more_info') NOT NULL,
    approved_amount DECIMAL(15, 2) NULL,
    approved_interest_rate DECIMAL(5, 2) NULL,
    decision_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (application_id) REFERENCES loan_applications(application_id) ON DELETE CASCADE
) ENGINE=InnoDB;
