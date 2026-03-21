// Utility functions for the application

function formatDate(date) {
    return new Date(date).toLocaleString();
}

function showNotification(message, type = 'info') {
    // Implementation for notifications
    console.log(`[${type}] ${message}`);
}

function validateVitals(vitals) {
    const errors = [];
    
    if (vitals.age && (vitals.age < 0 || vitals.age > 150)) {
        errors.push('Age must be between 0 and 150');
    }
    
    if (vitals.heart_rate && (vitals.heart_rate < 30 || vitals.heart_rate > 200)) {
        errors.push('Heart rate must be between 30 and 200 bpm');
    }
    
    if (vitals.oxygen && (vitals.oxygen < 50 || vitals.oxygen > 100)) {
        errors.push('Oxygen saturation must be between 50% and 100%');
    }
    
    if (vitals.temperature && (vitals.temperature < 30 || vitals.temperature > 45)) {
        errors.push('Temperature must be between 30°C and 45°C');
    }
    
    if (vitals.pain_level && (vitals.pain_level < 0 || vitals.pain_level > 10)) {
        errors.push('Pain level must be between 0 and 10');
    }
    
    return errors;
}