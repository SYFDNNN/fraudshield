/**
 * Simulation Playground JavaScript
 * Handles form interactions, preset loading, and prediction submission
 */

const PRESET_DESCRIPTIONS = {
    'rendah': 'Sinyal identitas dan riwayat relatif stabil. Preset hanya contoh input; kategori akhir tetap dihitung model.',
    'menengah': 'Profil campuran untuk demonstrasi awal. Preset hanya contoh input; kategori akhir tetap dihitung model.',
    'tinggi': 'Beberapa sinyal sistem memerlukan perhatian lebih. Preset hanya contoh input; kategori akhir tetap dihitung model.'
};

const RISK_LABELS = {
    'sangat_tinggi': 'Sangat tinggi',
    'tinggi': 'Tinggi',
    'menengah': 'Menengah',
    'rendah': 'Rendah'
};

// System signal field definitions
const SYSTEM_FIELDS = {
    'name_email_similarity': { label: 'Kemiripan nama–email', type: 'range', min: 0, max: 1, step: 0.01 },
    'prev_address_months_count': { label: 'Lama di alamat sebelumnya (bulan)', type: 'number', min: -1, step: 1 },
    'payment_type': { label: 'Jenis pembayaran', type: 'select', options: ['AA', 'AB', 'AC', 'AD', 'AE'] },
    'zip_count_4w': { label: 'ZIP count 4 minggu', type: 'number', min: 0, step: 1 },
    'velocity_6h': { label: 'Velocity 6 jam', type: 'number', min: 0, step: 1 },
    'velocity_24h': { label: 'Velocity 24 jam', type: 'number', min: 0, step: 1 },
    'velocity_4w': { label: 'Velocity 4 minggu', type: 'number', min: 0, step: 1 },
    'bank_branch_count_8w': { label: 'Bank branch count 8 minggu', type: 'number', min: 0, step: 1 },
    'date_of_birth_distinct_emails_4w': { label: 'DOB distinct emails 4 minggu', type: 'number', min: 0, step: 1 },
    'employment_status': { label: 'Status pekerjaan', type: 'select', options: ['CA', 'CB', 'CC', 'CD', 'CE', 'CF', 'CG'] },
    'email_is_free': { label: 'Email adalah free provider', type: 'select', options: [0, 1] },
    'housing_status': { label: 'Status tempat tinggal', type: 'select', options: ['BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG'] },
    'phone_home_valid': { label: 'Telepon rumah valid', type: 'select', options: [0, 1] },
    'bank_months_count': { label: 'Lama menjadi nasabah (bulan)', type: 'number', min: -1, step: 1 },
    'source': { label: 'Sumber aplikasi', type: 'select', options: ['INTERNET', 'TELEAPP'] },
    'session_length_in_minutes': { label: 'Durasi sesi (menit)', type: 'number', min: 0, step: 0.1 },
    'device_os': { label: 'Device OS', type: 'select', options: ['windows', 'macintosh', 'linux', 'x11', 'other'] },
    'keep_alive_session': { label: 'Keep alive session', type: 'select', options: [0, 1] },
    'device_distinct_emails_8w': { label: 'Device distinct emails 8 minggu', type: 'number', min: 0, step: 1 }
};

let currentPreset = 'rendah';
let experimentMode = false;
let presetData = {};

document.addEventListener('DOMContentLoaded', async () => {
    setupEventListeners();
    setupExpandable();
    await loadPresets();
    selectPreset('rendah');
});

function setupEventListeners() {
    // Preset buttons
    document.querySelectorAll('.fs-preset-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const preset = e.currentTarget.dataset.preset;
            selectPreset(preset);
        });
    });
    
    // Experiment mode toggle
    const toggle = document.getElementById('experimentToggle');
    toggle.addEventListener('click', () => {
        experimentMode = !experimentMode;
        toggle.classList.toggle('active');
        toggle.setAttribute('aria-checked', String(experimentMode));
        updateSystemFieldsAccess();
    });
    
    // Income range output
    const incomeInput = document.getElementById('income');
    const incomeValue = document.getElementById('incomeValue');
    incomeInput.addEventListener('input', (e) => {
        incomeValue.textContent = e.target.value;
    });
    
    // Submit button
    document.getElementById('submitBtn').addEventListener('click', submitPrediction);
    
    // Reset button
    document.getElementById('resetBtn').addEventListener('click', () => {
        loadPreset(currentPreset);
    });
}

function setupExpandable() {
    const expandable = document.getElementById('systemSignalsExpand');
    const header = expandable.querySelector('.fs-expandable-header');
    
    header.addEventListener('click', () => {
        const isExpanded = expandable.classList.toggle('expanded');
        header.setAttribute('aria-expanded', String(isExpanded));
    });
}

async function loadPresets() {
    try {
        const rendah = await FraudShieldAPI.get('/presets/rendah');
        const menengah = await FraudShieldAPI.get('/presets/menengah');
        const tinggi = await FraudShieldAPI.get('/presets/tinggi');
        
        presetData = { rendah, menengah, tinggi };
    } catch (error) {
        console.error('Failed to load presets:', error);
        showAlert('error', 'Gagal memuat preset data. Menggunakan default values.', 'alertContainer');
    }
}

function selectPreset(preset) {
    currentPreset = preset;
    
    // Update button states
    document.querySelectorAll('.fs-preset-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.preset === preset);
    });
    
    // Update description
    document.getElementById('presetDescription').textContent = PRESET_DESCRIPTIONS[preset];
    
    // Load preset values
    loadPreset(preset);
}

function loadPreset(preset) {
    const data = presetData[preset];
    if (!data) return;
    
    // Load main editable fields
    document.getElementById('applicationId').value = data.application_id ?? '';
    document.getElementById('income').value = data.income ?? 0.6;
    document.getElementById('incomeValue').textContent = data.income ?? 0.6;
    document.getElementById('customerAge').value = data.customer_age ?? 40;
    document.getElementById('currentAddressMonths').value = data.current_address_months_count ?? 24;
    document.getElementById('intendedBalconAmount').value = data.intended_balcon_amount ?? 20;
    document.getElementById('hasOtherCards').value = data.has_other_cards ?? 0;
    document.getElementById('proposedCreditLimit').value = data.proposed_credit_limit ?? 1000;
    document.getElementById('phoneMobileValid').value = data.phone_mobile_valid ?? 1;
    document.getElementById('foreignRequest').value = data.foreign_request ?? 0;
    
    // Render system fields
    renderSystemFields(data);
}

function renderSystemFields(data) {
    const container = document.getElementById('systemFieldsContainer');
    const disabled = !experimentMode;
    
    let html = '<div class="fs-form-grid">';
    
    for (const [fieldName, config] of Object.entries(SYSTEM_FIELDS)) {
        const value = data[fieldName] ?? '';
        const fieldId = `system_${fieldName}`;
        
        html += '<div class="fs-form-group">';
        html += `<label class="fs-label" for="${fieldId}">${escapeHtml(config.label)}</label>`;
        
        if (config.type === 'range') {
            html += `<input type="range" class="fs-input" id="${fieldId}" `;
            html += `min="${config.min}" max="${config.max}" step="${config.step}" `;
            html += `value="${value}" ${disabled ? 'disabled' : ''}>`;
            html += `<output id="${fieldId}_value" class="fs-text-small fs-text-muted">${value}</output>`;
        } else if (config.type === 'select') {
            html += `<select class="fs-select" id="${fieldId}" ${disabled ? 'disabled' : ''}>`;
            for (const option of config.options) {
                const selected = String(value) === String(option) ? 'selected' : '';
                const label = option === 0 ? 'Tidak' : option === 1 ? 'Ya' : option;
                html += `<option value="${option}" ${selected}>${escapeHtml(String(label))}</option>`;
            }
            html += '</select>';
        } else {
            html += `<input type="${config.type}" class="fs-input" id="${fieldId}" `;
            html += `min="${config.min}" step="${config.step}" value="${value}" ${disabled ? 'disabled' : ''}>`;
        }
        
        html += '</div>';
    }
    
    html += '</div>';
    container.innerHTML = html;
    
    // Setup range outputs
    for (const fieldName of Object.keys(SYSTEM_FIELDS)) {
        const field = SYSTEM_FIELDS[fieldName];
        if (field.type === 'range') {
            const input = document.getElementById(`system_${fieldName}`);
            const output = document.getElementById(`system_${fieldName}_value`);
            if (input && output) {
                input.addEventListener('input', (e) => {
                    output.textContent = e.target.value;
                });
            }
        }
    }
}

function updateSystemFieldsAccess() {
    const alert = document.getElementById('systemModeAlert');
    
    if (experimentMode) {
        alert.className = 'fs-alert fs-alert-warning';
        alert.textContent = 'Mode eksperimen aktif. Perubahan hanya untuk simulasi; pada produksi seluruh field berikut berasal dari sistem hulu.';
    } else {
        alert.className = 'fs-alert fs-alert-info';
        alert.textContent = 'Mode normal aktif: 19 field berikut read-only karena biasanya dihitung atau dipasok sistem bank. Masing-masing hanyalah sinyal statistik, bukan bukti kepalsuan.';
    }
    
    // Re-render fields with new access state
    const data = presetData[currentPreset];
    if (data) {
        renderSystemFields(data);
    }
}

function collectFormData() {
    const data = {
        application_id: document.getElementById('applicationId').value.trim(),
        income: parseFloat(document.getElementById('income').value),
        customer_age: parseInt(document.getElementById('customerAge').value),
        current_address_months_count: parseInt(document.getElementById('currentAddressMonths').value),
        intended_balcon_amount: parseFloat(document.getElementById('intendedBalconAmount').value),
        has_other_cards: parseInt(document.getElementById('hasOtherCards').value),
        proposed_credit_limit: parseFloat(document.getElementById('proposedCreditLimit').value),
        phone_mobile_valid: parseInt(document.getElementById('phoneMobileValid').value),
        foreign_request: parseInt(document.getElementById('foreignRequest').value)
    };
    
    // Collect system fields
    for (const fieldName of Object.keys(SYSTEM_FIELDS)) {
        const input = document.getElementById(`system_${fieldName}`);
        if (input) {
            const field = SYSTEM_FIELDS[fieldName];
            if (field.type === 'number' || field.type === 'range') {
                data[fieldName] = parseFloat(input.value);
            } else if (field.type === 'select') {
                const value = input.value;
                // Try to parse as number if it's 0 or 1
                data[fieldName] = (value === '0' || value === '1') ? parseInt(value) : value;
            } else {
                data[fieldName] = input.value;
            }
        }
    }
    
    return data;
}

async function submitPrediction() {
    clearAlert('alertContainer');
    
    const submitBtn = document.getElementById('submitBtn');
    const originalText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.dataset.loading = 'true';
    submitBtn.setAttribute('aria-busy', 'true');
    submitBtn.textContent = 'Memproses...';
    
    try {
        const formData = collectFormData();
        
        // Validate application_id
        if (!formData.application_id) {
            throw new Error('Application ID tidak boleh kosong');
        }
        
        const response = await FraudShieldAPI.post('/predict', formData);
        
        // Save to session storage
        SessionStore.set('fs_single_response', response);
        
        // Display result
        displayResult(response.prediction);
        
        showAlert('success', 'Prediksi berhasil dijalankan.', 'alertContainer');
        
    } catch (error) {
        console.error('Prediction failed:', error);
        showAlert('error', `Prediksi gagal: ${error.message}`, 'alertContainer');
        displayEmptyResult();
    } finally {
        submitBtn.disabled = false;
        submitBtn.dataset.loading = 'false';
        submitBtn.removeAttribute('aria-busy');
        submitBtn.textContent = originalText;
    }
}

function displayResult(prediction) {
    const container = document.getElementById('resultContent');
    const probability = Math.max(0, Math.min(1, prediction.fraud_probability || 0));
    const riskBand = prediction.risk_band || 'tidak_diketahui';
    const riskLabel = RISK_LABELS[riskBand] || riskBand.replaceAll('_', ' ');
    const analystAction = prediction.analyst_action || {};
    const reasonCodes = prediction.reason_codes || [];
    
    let html = `
        <div class="fs-probability-display">
            <div class="fs-probability-value">${formatProbability(probability)}</div>
            <div class="fs-risk-badge fs-risk-${riskBand}">${escapeHtml(riskLabel)}</div>
        </div>
        
        <div class="fs-result-section">
            <div class="fs-result-label">Application ID</div>
            <div class="fs-result-value fs-font-mono">${escapeHtml(prediction.application_id || '—')}</div>
        </div>
        
        <div class="fs-result-section">
            <div class="fs-result-label">Analyst Action</div>
            <div class="fs-result-value">${escapeHtml(analystAction.label || '—')}</div>
        </div>
        
        <div class="fs-result-section">
            <div class="fs-result-label">Review Recommended</div>
            <div class="fs-result-value">
                ${prediction.exact_capacity_review === true ? 'Ya' : prediction.exact_capacity_review === false ? 'Tidak' : 'N/A (single mode)'}
            </div>
        </div>
    `;
    
    if (reasonCodes.length > 0) {
        html += '<div class="fs-result-section">';
        html += '<div class="fs-result-label">Top Reason Codes</div>';
        html += '<ul class="fs-reason-list">';
        
        for (const reason of reasonCodes.slice(0, 5)) {
            const direction = reason.direction === 'increases_risk' ? 'increases' : 'decreases';
            const arrow = direction === 'increases' ? '↑' : '↓';
            const label = reason.feature_label || reason.feature || 'Unknown';
            const numericContribution = Number(reason.shap_contribution);
            const contribution = Number.isFinite(numericContribution)
                ? numericContribution.toFixed(4)
                : '—';
            
            html += `
                <li class="fs-reason-item">
                    <span class="fs-reason-direction ${direction}">${arrow}</span>
                    <span class="fs-reason-feature">${escapeHtml(label)}</span>
                    <span class="fs-reason-contribution">${contribution}</span>
                </li>
            `;
        }
        
        html += '</ul></div>';
    }
    
    container.innerHTML = html;
}

function displayEmptyResult() {
    const container = document.getElementById('resultContent');
    container.innerHTML = `
        <div class="fs-result-empty">
            <span class="fs-result-orbit" aria-hidden="true"></span>
            <div>
                <p><strong>Prediksi belum dapat ditampilkan</strong></p>
                <p>Periksa nilai input dan status koneksi API.</p>
            </div>
        </div>
    `;
}
