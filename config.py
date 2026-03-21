import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = True
    MODEL_PATH = 'model/'
    DATA_PATH = 'synthetic_medical_triage.csv'
    
class ProductionConfig(Config):
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'production-secret-key'
    
class DevelopmentConfig(Config):
    DEBUG = True