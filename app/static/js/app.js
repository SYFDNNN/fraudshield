/**
 * FraudShield Flask UI - Core JavaScript utilities
 * Handles API communication, UI updates, and interactions
 */

// API client
const FraudShieldAPI = {
    baseUrl: '/api',
    
    async get(endpoint) {
        const response = await fetch(`${this.baseUrl}${endpoint}`);
        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Network error' }));
            throw new Error(readErrorMessage(error, `HTTP ${response.status}`));
        }
        return response.json();
    },
    
    async post(endpoint, data) {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Network error' }));
            throw new Error(readErrorMessage(error, `HTTP ${response.status}`));
        }
        return response.json();
    }
};

// Global state
const AppState = {
    apiStatus: null,
    health: null,
    contract: null,
    
    setApiStatus(status, health = null, contract = null) {
        this.apiStatus = status;
        this.health = health;
        this.contract = contract;
        updateAPIStatusUI();
    }
};

// Initialize app on page load
document.addEventListener('DOMContentLoaded', () => {
    checkAPIStatus();
    setupAPIModal();
    setupMobileNavigation();
});

// API Status checking
async function checkAPIStatus() {
    const statusDot = document.getElementById('statusDot');
    const statusLabel = document.getElementById('statusLabel');
    const statusButton = document.getElementById('apiStatusBtn');

    if (!statusDot || !statusLabel || !statusButton) return;

    statusButton.setAttribute('aria-busy', 'true');
    statusButton.dataset.state = 'checking';
    
    try {
        const response = await FraudShieldAPI.get('/health');
        
        if (response.status === 'ready') {
            AppState.setApiStatus('online', response.health, response.contract);
            statusDot.classList.add('online');
            statusDot.classList.remove('offline');
            statusLabel.textContent = 'API ready';
            statusButton.dataset.state = 'online';
        } else {
            throw new Error('API not ready');
        }
    } catch (error) {
        AppState.setApiStatus('offline');
        statusDot.classList.add('offline');
        statusDot.classList.remove('online');
        statusLabel.textContent = 'API offline';
        statusButton.dataset.state = 'offline';
    } finally {
        statusButton.removeAttribute('aria-busy');
    }
}

// Update API status UI across the app
function updateAPIStatusUI() {
    // Dispatch custom event for components to listen
    const event = new CustomEvent('apiStatusChanged', {
        detail: AppState
    });
    document.dispatchEvent(event);
}

// Setup API status modal
function setupAPIModal() {
    const btn = document.getElementById('apiStatusBtn');
    if (btn) {
        btn.addEventListener('click', openApiModal);
    }

    document.querySelectorAll('[data-modal-close]').forEach((element) => {
        element.addEventListener('click', closeApiModal);
    });

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') closeApiModal();
    });
}

function openApiModal() {
    const modal = document.getElementById('apiModal');
    const content = document.getElementById('apiStatusContent');

    if (!modal || !content) return;

    modal.style.display = 'flex';
    modal.setAttribute('aria-hidden', 'false');
    document.body.classList.add('fs-modal-open');
    
    if (AppState.apiStatus === 'online' && AppState.health) {
        const reviewRate = AppState.contract?.requested_review_rate || 0;
        const modelVersion = AppState.health.model_version || 'Unknown';
        const calibrator = AppState.health.calibrator || 'Unknown';
        
        content.innerHTML = `
            <div class="fs-alert fs-alert-success">
                Ready endpoint terverifikasi
            </div>
            <div class="fs-form-group">
                <label class="fs-label">Model Version</label>
                <p class="fs-text-muted fs-text-small">${escapeHtml(modelVersion)}</p>
            </div>
            <div class="fs-form-group">
                <label class="fs-label">Calibrator</label>
                <p class="fs-text-muted fs-text-small">${escapeHtml(calibrator)}</p>
            </div>
            <div class="fs-form-group">
                <label class="fs-label">Review Cap</label>
                <p class="fs-text-muted fs-text-small">${(reviewRate * 100).toFixed(0)}%</p>
            </div>
            <button class="fs-btn fs-btn-secondary" onclick="checkAPIStatus(); setTimeout(openApiModal, 500)">
                Periksa ulang koneksi
            </button>
        `;
    } else {
        content.innerHTML = `
            <div class="fs-alert fs-alert-error">
                API tidak dapat dihubungi. Pastikan layanan aktif.
            </div>
            <div class="fs-form-group">
                <label class="fs-label">Troubleshooting</label>
                <ul class="fs-troubleshooting-list">
                    <li>Pastikan FastAPI berjalan di port 8000</li>
                    <li>Periksa environment variable FRAUDSHIELD_API_URL</li>
                    <li>Lihat console untuk error detail</li>
                </ul>
            </div>
            <button class="fs-btn fs-btn-secondary" onclick="checkAPIStatus(); setTimeout(openApiModal, 500)">
                Periksa ulang koneksi
            </button>
        `;
    }

    modal.querySelector('.fs-modal-close')?.focus();
}

function closeApiModal() {
    const modal = document.getElementById('apiModal');
    if (!modal || modal.style.display === 'none') return;
    modal.style.display = 'none';
    modal.setAttribute('aria-hidden', 'true');
    document.body.classList.remove('fs-modal-open');
    document.getElementById('apiStatusBtn')?.focus();
}

function setupMobileNavigation() {
    const topbar = document.getElementById('topbar');
    const toggle = document.getElementById('menuToggle');
    const nav = document.getElementById('primaryNav');

    if (!topbar || !toggle || !nav) return;

    const setOpen = (isOpen) => {
        topbar.classList.toggle('nav-open', isOpen);
        toggle.setAttribute('aria-expanded', String(isOpen));
        toggle.setAttribute('aria-label', isOpen ? 'Tutup navigasi' : 'Buka navigasi');
    };

    toggle.addEventListener('click', () => {
        setOpen(!topbar.classList.contains('nav-open'));
    });

    nav.querySelectorAll('a').forEach((link) => {
        link.addEventListener('click', () => setOpen(false));
    });

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') setOpen(false);
    });
}

function readErrorMessage(payload, fallback) {
    if (!payload || typeof payload !== 'object') return fallback;

    if (typeof payload.error === 'string') return payload.error;
    if (typeof payload.detail === 'string') return payload.detail;
    if (payload.detail && typeof payload.detail === 'object') {
        return payload.detail.message || payload.detail.error || fallback;
    }
    return fallback;
}

// Utility functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatProbability(prob) {
    return `${(prob * 100).toFixed(2)}%`;
}

function formatNumber(num) {
    return new Intl.NumberFormat('id-ID').format(num);
}

function showAlert(type, message, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const alert = document.createElement('div');
    alert.className = `fs-alert fs-alert-${type}`;
    alert.textContent = message;
    
    container.innerHTML = '';
    container.appendChild(alert);
}

function clearAlert(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = '';
    }
}

function showLoading(containerId, message = 'Loading...') {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const loading = document.createElement('div');
    loading.className = 'fs-loading';
    loading.textContent = message;
    
    container.innerHTML = '';
    container.appendChild(loading);
}

// Session storage helpers for persisting data
const SessionStore = {
    set(key, value) {
        try {
            sessionStorage.setItem(key, JSON.stringify(value));
        } catch (e) {
            console.warn('Failed to save to session storage:', e);
        }
    },
    
    get(key) {
        try {
            const item = sessionStorage.getItem(key);
            return item ? JSON.parse(item) : null;
        } catch (e) {
            console.warn('Failed to read from session storage:', e);
            return null;
        }
    },
    
    remove(key) {
        try {
            sessionStorage.removeItem(key);
        } catch (e) {
            console.warn('Failed to remove from session storage:', e);
        }
    }
};

// Export for use in other modules
window.FraudShieldAPI = FraudShieldAPI;
window.AppState = AppState;
window.SessionStore = SessionStore;
window.showAlert = showAlert;
window.clearAlert = clearAlert;
window.showLoading = showLoading;
window.formatProbability = formatProbability;
window.formatNumber = formatNumber;
window.escapeHtml = escapeHtml;
window.closeApiModal = closeApiModal;
