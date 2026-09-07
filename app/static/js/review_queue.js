/**
 * Review Queue JavaScript
 * Handles batch upload, validation, and results display
 */

const RISK_LABELS = {
    'sangat_tinggi': 'Sangat tinggi',
    'tinggi': 'Tinggi',
    'menengah': 'Menengah',
    'rendah': 'Rendah'
};

let uploadedData = null;
let validationResult = null;
let batchResults = null;

document.addEventListener('DOMContentLoaded', () => {
    setupUploadArea();
    setupTabs();
    setupSearch();
    setupExport();
    loadSessionData();
});

function setupUploadArea() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const clearBtn = document.getElementById('clearUploadBtn');
    const submitBtn = document.getElementById('submitBatchBtn');
    
    // Click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    uploadArea.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            fileInput.click();
        }
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
    
    // Clear button
    clearBtn.addEventListener('click', clearUpload);
    
    // Submit button
    submitBtn.addEventListener('click', submitBatch);
}

async function handleFileUpload(file) {
    clearAlert('uploadAlertContainer');
    
    if (!file.name.endsWith('.json')) {
        showAlert('error', 'File harus berformat JSON', 'uploadAlertContainer');
        return;
    }
    
    if (file.size > 16 * 1024 * 1024) {
        showAlert('error', 'File terlalu besar. Batas maksimum 16 MB.', 'uploadAlertContainer');
        return;
    }
    
    try {
        const text = await file.text();
        const data = JSON.parse(text);
        
        uploadedData = data;
        await validateBatch(data);
        
    } catch (error) {
        console.error('File upload error:', error);
        showAlert('error', `Gagal memproses file: ${error.message}`, 'uploadAlertContainer');
    }
}

async function validateBatch(data) {
    const batchIdInput = document.getElementById('batchIdInput');
    const fallbackBatchId = batchIdInput.value.trim() || 'batch-upload-001';
    
    try {
        const response = await FraudShieldAPI.post('/validate-batch', {
            payload: data,
            batch_id: fallbackBatchId
        });
        
        validationResult = response;
        displayValidation(response);
        
    } catch (error) {
        console.error('Validation error:', error);
        showAlert('error', `Validasi gagal: ${error.message}`, 'uploadAlertContainer');
    }
}

function displayValidation(result) {
    const section = document.getElementById('validationSection');
    const summary = document.getElementById('validationSummary');
    const issues = document.getElementById('validationIssues');
    const submitBtn = document.getElementById('submitBatchBtn');
    
    section.classList.add('show');
    
    // Summary stats
    summary.innerHTML = `
        <div class="fs-validation-stat">
            <div class="fs-validation-stat-label">Total Rows</div>
            <div class="fs-validation-stat-value">${formatNumber(result.row_count)}</div>
        </div>
        <div class="fs-validation-stat">
            <div class="fs-validation-stat-label">Unique IDs</div>
            <div class="fs-validation-stat-value">${formatNumber(result.unique_id_count)}</div>
        </div>
        <div class="fs-validation-stat">
            <div class="fs-validation-stat-label">Batch ID</div>
            <div class="fs-validation-stat-value fs-validation-stat-value--compact">
                ${escapeHtml(result.batch_id)}
            </div>
        </div>
        <div class="fs-validation-stat">
            <div class="fs-validation-stat-label">Status</div>
            <div class="fs-validation-stat-value fs-validation-stat-value--status ${result.valid ? 'status-online' : 'status-offline'}">
                ${result.valid ? 'Valid' : 'Tidak valid'}
            </div>
        </div>
    `;
    
    // Issues
    let issuesHtml = '';
    
    if (result.errors && result.errors.length > 0) {
        issuesHtml += '<div class="fs-alert fs-alert-error fs-validation-issues">';
        issuesHtml += '<strong>Errors:</strong>';
        issuesHtml += '<ul>';
        for (const error of result.errors) {
            issuesHtml += `<li>${escapeHtml(error)}</li>`;
        }
        issuesHtml += '</ul></div>';
    }
    
    if (result.warnings && result.warnings.length > 0) {
        issuesHtml += '<div class="fs-alert fs-alert-warning fs-validation-issues fs-validation-warnings">';
        issuesHtml += '<strong>Warnings:</strong>';
        issuesHtml += '<ul>';
        for (const warning of result.warnings) {
            issuesHtml += `<li>${escapeHtml(warning)}</li>`;
        }
        issuesHtml += '</ul></div>';
    }
    
    if (!issuesHtml && result.valid) {
        issuesHtml = '<div class="fs-alert fs-alert-success">Batch valid dan siap membentuk antrean review.</div>';
    }
    
    issues.innerHTML = issuesHtml;
    
    // Enable/disable submit button
    submitBtn.disabled = !result.valid;
}

async function submitBatch() {
    if (!uploadedData || !validationResult || !validationResult.valid) {
        return;
    }
    
    clearAlert('uploadAlertContainer');
    
    const submitBtn = document.getElementById('submitBatchBtn');
    const originalText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.dataset.loading = 'true';
    submitBtn.setAttribute('aria-busy', 'true');
    submitBtn.textContent = 'Memproses...';
    
    try {
        // Prepare payload
        let payload;
        if (Array.isArray(uploadedData)) {
            payload = {
                batch_id: validationResult.batch_id,
                applications: uploadedData
            };
        } else {
            payload = uploadedData;
            if (!payload.batch_id) {
                payload.batch_id = validationResult.batch_id;
            }
        }
        
        const response = await FraudShieldAPI.post('/predict/batch', payload);
        
        batchResults = response;
        
        // Save to session
        SessionStore.set('fs_batch_response', response);
        
        // Display results
        displayResults(response);
        
        showAlert('success', 'Antrean review berhasil dibentuk.', 'uploadAlertContainer');
        
    } catch (error) {
        console.error('Batch prediction error:', error);
        showAlert('error', `Batch gagal: ${error.message}`, 'uploadAlertContainer');
    } finally {
        submitBtn.disabled = false;
        submitBtn.dataset.loading = 'false';
        submitBtn.removeAttribute('aria-busy');
        submitBtn.textContent = originalText;
    }
}

function clearUpload() {
    uploadedData = null;
    validationResult = null;
    
    document.getElementById('fileInput').value = '';
    document.getElementById('validationSection').classList.remove('show');
    clearAlert('uploadAlertContainer');
}

function displayResults(response) {
    const section = document.getElementById('resultsSection');
    const predictions = Array.isArray(response.predictions) ? response.predictions : [];
    section.classList.add('show');
    
    // Update header
    document.getElementById('resultsBatchId').textContent = response.batch_id || 'Hasil batch';
    document.getElementById('resultsRowCount').textContent = formatNumber(response.row_count ?? predictions.length);
    document.getElementById('resultsReviewCount').textContent = formatNumber(response.review_count ?? 0);
    
    const reviewRate = response.requested_review_rate || 0;
    document.getElementById('resultsReviewRate').textContent = `${(reviewRate * 100).toFixed(0)}%`;
    document.getElementById('resultsContractVersion').textContent = response.contract_version || '—';
    
    // Risk distribution
    displayRiskDistribution(predictions);
    
    // Update tab counts
    const reviewQueue = predictions.filter(p => p.exact_capacity_review === true);
    document.getElementById('reviewQueueCount').textContent = formatNumber(reviewQueue.length);
    document.getElementById('allAppsCount').textContent = formatNumber(predictions.length);
    
    // Populate tables
    populateReviewTable(reviewQueue);
    populateAllTable(predictions);
    
    // Scroll to results
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayRiskDistribution(predictions) {
    const container = document.getElementById('riskBars');
    const total = predictions.length;
    
    // Count by risk band
    const counts = {
        'sangat_tinggi': 0,
        'tinggi': 0,
        'menengah': 0,
        'rendah': 0
    };
    
    for (const pred of predictions) {
        const band = pred.risk_band || 'rendah';
        if (counts.hasOwnProperty(band)) {
            counts[band]++;
        }
    }
    
    // Render bars
    let html = '';
    for (const [band, count] of Object.entries(counts)) {
        const percentage = total > 0 ? (count / total) * 100 : 0;
        html += `
            <div class="fs-risk-bar-row">
                <div class="fs-risk-bar-label">${RISK_LABELS[band]}</div>
                <div class="fs-risk-bar-track">
                    <div class="fs-risk-bar-fill fs-risk-bar-${band}" 
                         style="width: ${percentage}%"></div>
                </div>
                <div class="fs-risk-bar-value">
                    ${count} (${percentage.toFixed(1)}%)
                </div>
            </div>
        `;
    }
    
    container.innerHTML = html;
}

function populateReviewTable(predictions) {
    const tbody = document.getElementById('reviewTableBody');
    
    if (predictions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="fs-table-empty">Tidak ada pengajuan dalam antrean review.</td></tr>';
        return;
    }
    
    let html = '';
    predictions.forEach((pred, index) => {
        const reasons = renderReasonSummary(pred.reason_codes);
        const riskBand = pred.risk_band || 'rendah';
        const riskLabel = RISK_LABELS[riskBand] || riskBand.replaceAll('_', ' ');
        const analystAction = pred.analyst_action || {};
        
        html += `
            <tr>
                <td>${index + 1}</td>
                <td class="fs-table-id">${escapeHtml(pred.application_id)}</td>
                <td class="fs-table-probability">${formatProbability(pred.fraud_probability)}</td>
                <td>
                    <span class="fs-table-risk-badge fs-risk-badge fs-risk-${riskBand}">
                        ${escapeHtml(riskLabel)}
                    </span>
                </td>
                <td class="fs-table-action-cell">${escapeHtml(analystAction.label || '—')}</td>
                <td class="fs-table-reasons">${reasons}</td>
            </tr>
        `;
    });
    
    tbody.innerHTML = html;
}

function populateAllTable(predictions) {
    const tbody = document.getElementById('allTableBody');
    
    if (predictions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="fs-table-empty">Belum ada data pengajuan.</td></tr>';
        return;
    }
    
    let html = '';
    predictions.forEach((pred) => {
        const reasons = renderReasonSummary(pred.reason_codes);
        const riskBand = pred.risk_band || 'rendah';
        const riskLabel = RISK_LABELS[riskBand] || riskBand.replaceAll('_', ' ');
        const analystAction = pred.analyst_action || {};
        const review = pred.exact_capacity_review;
        const reviewClass = review ? 'fs-table-review-yes' : 'fs-table-review-no';
        const reviewText = review === true ? 'Ya' : review === false ? 'Tidak' : '—';
        
        html += `
            <tr>
                <td class="fs-table-id">${escapeHtml(pred.application_id)}</td>
                <td class="fs-table-probability">${formatProbability(pred.fraud_probability)}</td>
                <td>
                    <span class="fs-table-risk-badge fs-risk-badge fs-risk-${riskBand}">
                        ${escapeHtml(riskLabel)}
                    </span>
                </td>
                <td class="${reviewClass}">${reviewText}</td>
                <td class="fs-table-action-cell">${escapeHtml(analystAction.label || '—')}</td>
                <td class="fs-table-reasons">${reasons}</td>
            </tr>
        `;
    });
    
    tbody.innerHTML = html;
}

function getReasonSummary(reasonCodes, limit = 3) {
    if (!Array.isArray(reasonCodes) || reasonCodes.length === 0) {
        return 'Alasan lokal belum tersedia';
    }
    
    const labels = [];
    for (const reason of reasonCodes.slice(0, limit)) {
        const label = reason.feature_label || reason.feature || 'Unknown';
        const arrow = reason.direction === 'increases_risk' ? '↑' : '↓';
        labels.push(`${arrow} ${label}`);
    }
    
    return labels.join('; ');
}

function renderReasonSummary(reasonCodes, limit = 3) {
    if (!Array.isArray(reasonCodes) || reasonCodes.length === 0) {
        return '<span class="fs-reason-empty">Alasan lokal belum tersedia</span>';
    }

    const rows = reasonCodes.slice(0, limit).map((reason) => {
        const label = reason.feature_label || reason.feature || 'Unknown';
        const increasesRisk = reason.direction === 'increases_risk';
        const arrow = increasesRisk ? '↑' : '↓';
        const arrowClass = increasesRisk
            ? 'fs-reason-arrow--up'
            : 'fs-reason-arrow--down';

        return `
            <span class="fs-reason-line">
                <span class="fs-reason-arrow ${arrowClass}" aria-hidden="true">${arrow}</span>
                <span>${escapeHtml(label)}</span>
            </span>
        `;
    }).join('');

    return `<span class="fs-reason-summary">${rows}</span>`;
}

function setupTabs() {
    document.querySelectorAll('.fs-queue-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            const tabName = e.currentTarget.dataset.tab;
            switchTab(tabName);
        });
    });
}

function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.fs-queue-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });
    
    // Update tab content
    document.querySelectorAll('.fs-tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `${tabName}Tab`);
    });
}

function setupSearch() {
    const reviewSearch = document.getElementById('reviewSearchInput');
    const allSearch = document.getElementById('allSearchInput');
    
    reviewSearch.addEventListener('input', (e) => {
        filterTable('reviewTableBody', e.target.value);
    });
    
    allSearch.addEventListener('input', (e) => {
        filterTable('allTableBody', e.target.value);
    });
}

function filterTable(tbodyId, searchTerm) {
    const tbody = document.getElementById(tbodyId);
    const rows = tbody.querySelectorAll('tr');
    const term = searchTerm.toLowerCase();
    
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(term) ? '' : 'none';
    });
}

function setupExport() {
    document.getElementById('exportReviewCsv').addEventListener('click', () => {
        exportToCSV('review');
    });
    
    document.getElementById('exportAllCsv').addEventListener('click', () => {
        exportToCSV('all');
    });
}

function exportToCSV(type) {
    if (!batchResults || !batchResults.predictions) {
        showAlert('warning', 'Tidak ada data untuk di-export', 'uploadAlertContainer');
        return;
    }
    
    const predictions = type === 'review' 
        ? batchResults.predictions.filter(p => p.exact_capacity_review === true)
        : batchResults.predictions;
    
    if (predictions.length === 0) {
        showAlert('warning', 'Tidak ada data untuk di-export', 'uploadAlertContainer');
        return;
    }
    
    // CSV headers
    const headers = [
        'application_id',
        'fraud_probability',
        'risk_band',
        'exact_capacity_review',
        'analyst_action_code',
        'analyst_action_label',
        'top_reasons'
    ];
    
    // CSV rows
    const rows = predictions.map(pred => {
        const analystAction = pred.analyst_action || {};
        return [
            pred.application_id,
            pred.fraud_probability,
            pred.risk_band,
            pred.exact_capacity_review,
            analystAction.code || '',
            analystAction.label || '',
            getReasonSummary(pred.reason_codes, 5)
        ].map(v => `"${String(v).replace(/"/g, '""')}"`).join(',');
    });
    
    const csv = [headers.join(','), ...rows].join('\n');
    
    // Download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fraudshield_${type}_${batchResults.batch_id}_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
}

function loadSessionData() {
    const savedBatch = SessionStore.get('fs_batch_response');
    if (savedBatch) {
        batchResults = savedBatch;
        displayResults(savedBatch);
    }
}
