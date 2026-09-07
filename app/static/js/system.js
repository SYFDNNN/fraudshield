/**
 * Model & System page JavaScript
 * Displays contract information and system details
 */

document.addEventListener('DOMContentLoaded', () => {
    loadContractInfo();
    setupExampleTabs();
    setupCopyButtons();
});

async function loadContractInfo() {
    try {
        const health = await FraudShieldAPI.get('/health');
        
        if (health.status !== 'ready') {
            showAlert('error', 'API tidak tersedia. Koneksi ke API diperlukan untuk melihat contract info.', 'systemAlertContainer');
            return;
        }
        
        const contract = await FraudShieldAPI.get('/contract');
        
        displayContractInfo(health, contract);
        
    } catch (error) {
        console.error('Failed to load contract:', error);
        showAlert('error', `Gagal memuat contract info: ${error.message}`, 'systemAlertContainer');
    }
}

function displayContractInfo(health, contract) {
    // Header
    document.getElementById('contractVersion').textContent = 
        contract.contract_version || '—';
    
    // Meta
    document.getElementById('modelVersion').textContent = 
        health.health?.model_version || '—';
    document.getElementById('calibratorName').textContent = 
        health.health?.calibrator || '—';
    
    const reviewRate = contract.requested_review_rate || 0;
    document.getElementById('reviewCapacity').textContent = 
        `${(reviewRate * 100).toFixed(0)}%`;
    
    document.getElementById('maxBatchSize').textContent = 
        formatNumber(contract.maximum_batch_size || 5000);
    
    // Required fields
    displayFieldList(
        'requiredFieldsList',
        contract.required_features || [],
        'required'
    );
    
    // Forbidden fields
    displayFieldList(
        'forbiddenFieldsList',
        contract.forbidden_features || [],
        'forbidden'
    );
    
    // Missing semantics
    displayMissingSemantics(contract.missing_value_semantics || {});
}

function displayFieldList(containerId, fields, type) {
    const container = document.getElementById(containerId);
    
    if (fields.length === 0) {
        container.innerHTML = '<p class="fs-text-muted">Belum ada field yang didefinisikan.</p>';
        return;
    }
    
    let html = '';
    for (const field of fields) {
        const fieldName = typeof field === 'string' ? field : field.name;
        const fieldType = typeof field === 'object' ? field.type : 'any';
        const badgeClass = type === 'required' ? 'fs-field-badge-required' : 'fs-field-badge-forbidden';
        const badgeText = type === 'required' ? 'Wajib' : 'Dilarang';
        
        html += `
            <div class="fs-field-item">
                <div>
                    <div class="fs-field-name">${escapeHtml(fieldName)}</div>
                </div>
                <div class="fs-field-meta">
                    <span class="fs-field-type">${escapeHtml(fieldType)}</span>
                    <span class="fs-field-badge ${badgeClass}">${badgeText}</span>
                </div>
            </div>
        `;
    }
    
    container.innerHTML = html;
}

function displayMissingSemantics(semantics) {
    const container = document.getElementById('missingSemanticsGrid');
    
    const semanticsList = Object.entries(semantics);
    
    if (semanticsList.length === 0) {
        container.innerHTML = '<p class="fs-text-muted">Semantik missing value belum tersedia.</p>';
        return;
    }
    
    let html = '';
    for (const [fieldType, values] of semanticsList) {
        html += `
            <div class="fs-policy-item">
                <h4>${escapeHtml(fieldType)}</h4>
                <dl>
        `;
        
        if (Array.isArray(values)) {
            for (const value of values) {
                html += `
                    <div class="fs-policy-detail">
                        <dt>Sentinel Value</dt>
                        <dd>${escapeHtml(String(value))}</dd>
                    </div>
                `;
            }
        } else if (typeof values === 'object') {
            for (const [key, value] of Object.entries(values)) {
                html += `
                    <div class="fs-policy-detail">
                        <dt>${escapeHtml(key)}</dt>
                        <dd>${escapeHtml(String(value))}</dd>
                    </div>
                `;
            }
        } else {
            html += `
                <div class="fs-policy-detail">
                    <dt>Value</dt>
                    <dd>${escapeHtml(String(values))}</dd>
                </div>
            `;
        }
        
        html += '</dl></div>';
    }
    
    container.innerHTML = html;
}

function setupExampleTabs() {
    document.querySelectorAll('.fs-example-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            const exampleType = e.currentTarget.dataset.example;
            
            // Update tabs
            document.querySelectorAll('.fs-example-tab').forEach(t => {
                t.classList.toggle('active', t.dataset.example === exampleType);
            });
            
            // Update content
            document.querySelectorAll('.fs-example-content').forEach(content => {
                content.classList.toggle('active', content.id === `${exampleType}Example`);
            });
        });
    });
}

function setupCopyButtons() {
    document.querySelectorAll('.fs-copy-button').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const button = e.currentTarget;
            const codeId = button.dataset.copy;
            const codeElement = document.getElementById(codeId);
            
            if (codeElement) {
                const text = codeElement.textContent;
                
                navigator.clipboard.writeText(text).then(() => {
                    const originalText = button.textContent;
                    button.textContent = 'Tersalin';
                    button.classList.add('copied');
                    
                    setTimeout(() => {
                        button.textContent = originalText;
                        button.classList.remove('copied');
                    }, 2000);
                }).catch(err => {
                    console.error('Failed to copy:', err);
                });
            }
        });
    });
}
