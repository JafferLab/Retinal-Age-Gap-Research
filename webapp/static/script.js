document.addEventListener('DOMContentLoaded', () => {
    const predictForm = document.getElementById('predict-form');
    const gapForm = document.getElementById('gap-form');
    const fileInput = document.getElementById('fundus-image');
    const filePlaceholder = document.querySelector('.file-upload-placeholder');
    const loadingSpinner = document.getElementById('loading-spinner');
    const resultsSection = document.getElementById('results-section');
    const gapSection = document.getElementById('gap-section');
    const gapResults = document.getElementById('gap-results');

    let currentPredictedAge = null;

    // File input change handler
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            filePlaceholder.textContent = e.target.files[0].name;
        }
    });

    // Predict Form Submit
    predictForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(predictForm);

        // UI Updates
        predictForm.classList.add('hidden');
        loadingSpinner.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        gapSection.classList.add('hidden');
        gapResults.classList.add('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(await response.text());
            }

            const data = await response.json();
            currentPredictedAge = data.predicted_age;

            // Update Results
            document.getElementById('predicted-age').textContent = data.predicted_age.toFixed(1);

            const qcStatusEl = document.getElementById('qc-status');
            qcStatusEl.textContent = data.qc_status;

            // QC Styling
            qcStatusEl.className = 'value'; // Reset
            if (data.qc_status === 'PASS') {
                qcStatusEl.style.color = 'var(--success-color)';
                document.getElementById('qc-details').classList.add('hidden');
            } else if (data.qc_status === 'WARN') {
                qcStatusEl.style.color = 'var(--warning-color)';
                document.getElementById('qc-details').classList.remove('hidden');
                document.getElementById('qc-reasons').textContent = data.qc_reasons.join(', ');
            } else {
                qcStatusEl.style.color = 'var(--error-color)';
                document.getElementById('qc-details').classList.remove('hidden');
                document.getElementById('qc-reasons').textContent = data.qc_reasons.join(', ');
            }

            document.getElementById('model-version').textContent = data.model_version;
            document.getElementById('app-version').textContent = data.app_version;

            // Show Results & Next Step
            resultsSection.classList.remove('hidden');
            gapSection.classList.remove('hidden');

        } catch (error) {
            alert('Error: ' + error.message);
            predictForm.classList.remove('hidden'); // Show form again on error
        } finally {
            loadingSpinner.classList.add('hidden');
        }
    });

    // Gap Form Submit
    gapForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (currentPredictedAge === null) return;

        const formData = new FormData(gapForm);
        formData.append('predicted_age', currentPredictedAge);

        try {
            const response = await fetch('/calculate_gap', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(await response.text());
            }

            const data = await response.json();

            document.getElementById('chrono-age').textContent = data.chronological_age;

            const gapEl = document.getElementById('age-gap');
            const gap = data.retinal_age_gap;
            gapEl.textContent = (gap > 0 ? '+' : '') + gap.toFixed(1);

            // Color code gap
            if (Math.abs(gap) >= 3) {
                gapEl.style.color = gap > 0 ? 'var(--error-color)' : 'var(--success-color)'; // Older = bad? Just highlighting significant gaps
            } else {
                gapEl.style.color = 'var(--text-color)';
            }

            gapResults.classList.remove('hidden');

        } catch (error) {
            alert('Error: ' + error.message);
        }
    });

    // Start Over logic
    const startOverButtons = document.querySelectorAll('.start-over-btn');
    startOverButtons.forEach(btn => {
        btn.addEventListener('click', resetApp);
    });

    function resetApp() {
        // Reset forms
        predictForm.reset();
        gapForm.reset();

        // Hide results sections
        resultsSection.classList.add('hidden');
        gapSection.classList.add('hidden');
        gapResults.classList.add('hidden');

        // Show upload section
        uploadSection.classList.remove('hidden');

        // Clear results text
        predictedAgeVal.textContent = '--';
        qcStatus.textContent = '--';
        qcStatus.className = 'status-badge';
        qcIssues.textContent = '';
        chronoAgeVal.textContent = '--';
        ageGapVal.textContent = '--';

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
});
