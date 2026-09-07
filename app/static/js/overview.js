/**
 * Overview page JavaScript
 * Handles system status display and session activity
 */

document.addEventListener('DOMContentLoaded', () => {
    loadSystemStatus();
    loadSessionActivity();
    
    // Listen for API status changes
    document.addEventListener('apiStatusChanged', (event) => {
        updateSystemDisplay(event.detail);
    });
});

async function loadSystemStatus() {
    try {
        const response = await FraudShieldAPI.get('/health');
        
        if (response.status === 'ready') {
            updateSystemDisplay({
                apiStatus: 'online',
                health: response.health,
                contract: response.contract
            });
        } else {
            updateSystemDisplay({ apiStatus: 'offline' });
        }
    } catch (error) {
        console.error('Failed to load system status:', error);
        updateSystemDisplay({ apiStatus: 'offline' });
    }
}

function updateSystemDisplay(state) {
    const isOnline = state.apiStatus === 'online';
    const health = state.health || {};
    const contract = state.contract || {};
    
    const stageStatus = document.getElementById('stageApiStatus');
    const stageLabel = document.getElementById('stageApiLabel');

    if (stageStatus && stageLabel) {
        stageStatus.classList.toggle('fs-stage-online', isOnline);
        stageStatus.classList.toggle('fs-stage-offline', !isOnline);
        stageLabel.textContent = isOnline ? 'API ready' : 'API offline';
    }
    
    // API Status
    const apiLabel = document.getElementById('apiLabel');
    const apiNote = document.getElementById('apiNote');
    if (apiLabel && apiNote) {
        apiLabel.textContent = isOnline ? 'Operational' : 'Offline';
        apiLabel.className = 'value ' + (isOnline ? 'status-online' : 'status-offline');
        apiNote.textContent = isOnline ? 'Ready endpoint terverifikasi' : 'Jalankan layanan Uvicorn';
    }
    
    // Model Build
    const modelLabel = document.getElementById('modelLabel');
    const modelNote = document.getElementById('modelNote');
    if (modelLabel && modelNote) {
        modelLabel.textContent = health.model_version || 'Tidak tersedia';
        modelNote.textContent = health.calibrator ? `Kalibrasi · ${health.calibrator}` : 'Model belum dimuat';
    }
    
    // Contract
    const contractLabel = document.getElementById('contractLabel');
    if (contractLabel) {
        contractLabel.textContent = contract.contract_version || 'Tidak tersedia';
    }
    
    // Review Policy
    const reviewLabel = document.getElementById('reviewLabel');
    if (reviewLabel) {
        const hasContract = Object.keys(contract).length > 0;
        const reviewRate = contract.requested_review_rate || 0;
        reviewLabel.textContent = hasContract ? `${(reviewRate * 100).toFixed(0)}% capacity` : 'Belum tersedia';
    }
}

function loadSessionActivity() {
    // Load single prediction activity
    const singleResponse = SessionStore.get('fs_single_response');
    if (singleResponse && singleResponse.prediction) {
        displaySingleActivity(singleResponse.prediction);
    }
    
    // Load batch prediction activity
    const batchResponse = SessionStore.get('fs_batch_response');
    if (batchResponse) {
        displayBatchActivity(batchResponse);
    }
}

function displaySingleActivity(prediction) {
    const container = document.getElementById('singleActivity');
    if (!container) return;
    
    const probability = Math.max(0, Math.min(1, prediction.fraud_probability || 0));
    const riskBand = prediction.risk_band || '—';
    const riskLabels = {
        'sangat_tinggi': 'Sangat tinggi',
        'tinggi': 'Tinggi',
        'menengah': 'Menengah',
        'rendah': 'Rendah'
    };
    const riskLabel = riskLabels[riskBand] || riskBand.replace('_', ' ');
    
    const analystAction = prediction.analyst_action || {};
    const actionLabel = analystAction.label || 'Pemeriksaan standar';
    const applicationId = prediction.application_id || '—';
    
    container.className = 'fs-activity-result';
    container.innerHTML = `
        <div class="fs-activity-result-value">${formatProbability(probability)}</div>
        <div class="fs-activity-result-id">${escapeHtml(applicationId)}</div>
        <div class="fs-activity-result-meta">
            <span>${escapeHtml(riskLabel)}</span>
            <span>${escapeHtml(actionLabel)}</span>
        </div>
    `;
}

function displayBatchActivity(batchResponse) {
    const container = document.getElementById('batchActivity');
    if (!container) return;
    
    const rowCount = batchResponse.row_count || 0;
    const reviewCount = batchResponse.review_count || 0;
    const batchId = batchResponse.batch_id || '—';
    
    container.className = 'fs-activity-result';
    container.innerHTML = `
        <div class="fs-activity-result-value">
            ${formatNumber(reviewCount)} <small>/ ${formatNumber(rowCount)} pengajuan</small>
        </div>
        <div class="fs-activity-result-id">${escapeHtml(batchId)}</div>
        <div class="fs-activity-result-meta">
            <span>Masuk review</span>
            <span>Exact capacity</span>
        </div>
    `;
}
