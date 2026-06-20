const API_BASE = "/api";
let selectedFile = null;

document.addEventListener('DOMContentLoaded', function() {
  const fileInput = document.getElementById('fileInput');
  fileInput.addEventListener('change', handleFileSelect);
});

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (!file) return;

  const validTypes = ['image/png', 'image/jpeg', 'application/dicom', 'application/octet-stream'];
  const validExtensions = ['.jpg', '.jpeg', '.png', '.dcm'];

  const isValidType = validTypes.includes(file.type) ||
                      validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));

  if (!isValidType) {
    showError('Invalid file type. Please upload JPG, PNG, or DCM files only.');
    clearFileSelection();
    return;
  }

  selectedFile = file;

  document.getElementById('fileName').textContent = `Selected: ${file.name}`;

  const reader = new FileReader();
  reader.onload = function(e) {
    if (!file.name.toLowerCase().endsWith('.dcm')) {
      document.getElementById('original').src = e.target.result;
      document.getElementById('original').style.display = 'block';
    } else {
      document.getElementById('original').innerHTML = '<p>DICOM file selected - analyzing will show preview</p>';
    }
  };
  reader.readAsDataURL(file);
}

async function analyzeImage() {
  if (!selectedFile) {
    showError('Please select an image file first');
    return;
  }

  const formData = new FormData();
  formData.append('file', selectedFile);

  showLoading(true);
  hideResults();

  try {
    const response = await fetch(`${API_BASE}/detect-medical-tamper`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const error = await response.json();
      showError(error.detail || 'Analysis failed');
      showLoading(false);
      return;
    }

    const result = await response.json();
    showLoading(false);
    displayResults(result);
  } catch (error) {
    console.error('Error:', error);
    showError('Network error: ' + error.message);
    showLoading(false);
  }
}

function displayResults(result) {
  const classification = result.classification;
  const statusElement = document.getElementById('status');
  statusElement.textContent = classification;
  statusElement.className = 'status-text ' + classification.toLowerCase();

  const tamperedProb = (result.tampered_probability * 100).toFixed(1);
  const authenticProb = (result.authentic_probability * 100).toFixed(1);

  document.getElementById('tamperedProb').textContent = tamperedProb;
  document.getElementById('authenticProb').textContent = authenticProb;

  const confidence = classification === 'Tampered' ? tamperedProb : authenticProb;
  document.getElementById('confidence').innerHTML = `<strong>Confidence:</strong> ${confidence}%`;

  const riskLevel = (result.tampered_probability * 100).toFixed(0);
  updateGauge(riskLevel);

  const localizationSection = document.getElementById('localizationSection');
  const heatmapImage = document.getElementById('heatmapImage');

  if (classification === 'Tampered' && (result.heatmap || result.heatmap_path)) {
    localizationSection.style.display = 'block';

    if (result.heatmap) {
      displayHeatmap(result.heatmap);
    } else if (result.heatmap_path) {
      heatmapImage.src = result.heatmap_path;
    }
  } else {
    localizationSection.style.display = 'none';
    heatmapImage.removeAttribute('src');
  }

  document.getElementById('results').style.display = 'block';
  document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

function displayHeatmap(heatmapArray) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  const width = 224;
  const height = 224;
  canvas.width = width;
  canvas.height = height;

  const imageData = ctx.createImageData(width, height);
  const data = imageData.data;

  let dataIndex = 0;
  for (let i = 0; i < heatmapArray.length; i++) {
    const pixelArray = heatmapArray[i];
    for (let j = 0; j < pixelArray.length; j++) {
      const pixelValues = pixelArray[j];
      data[dataIndex] = Math.round(pixelValues[0] * 255);
      data[dataIndex + 1] = Math.round(pixelValues[1] * 255);
      data[dataIndex + 2] = Math.round(pixelValues[2] * 255);
      data[dataIndex + 3] = 255;
      dataIndex += 4;
    }
  }

  ctx.putImageData(imageData, 0, 0);

  const heatmapImage = document.getElementById('heatmapImage');
  heatmapImage.src = canvas.toDataURL();
}

function updateGauge(percent) {
  percent = Math.min(Math.max(percent, 0), 100);

  const fill = document.getElementById('gaugeFill');
  const text = document.getElementById('gaugeText');

  const rotation = (percent / 100) * 180 - 90;
  fill.style.transform = `rotate(${rotation}deg)`;

  text.textContent = percent + '%';

  if (percent < 30) {
    fill.style.background = 'linear-gradient(to top, #10B981, #3B82F6)';
    text.style.color = '#10B981';
  } else if (percent < 60) {
    fill.style.background = 'linear-gradient(to top, #F59E0B, #3B82F6)';
    text.style.color = '#F59E0B';
  } else {
    fill.style.background = 'linear-gradient(to top, #EF4444, #3B82F6)';
    text.style.color = '#EF4444';
  }
}

function showLoading(show) {
  document.getElementById('loading').style.display = show ? 'block' : 'none';
  const analyzeBtn = document.querySelector('button[onclick="analyzeImage()"]');
  if (analyzeBtn) {
    analyzeBtn.disabled = show;
  }
}

function hideResults() {
  document.getElementById('results').style.display = 'none';
}

function showError(message) {
  const existingError = document.querySelector('.error');
  if (existingError) {
    existingError.remove();
  }

  const errorDiv = document.createElement('div');
  errorDiv.className = 'error';
  errorDiv.textContent = '❌ ' + message;

  const container = document.querySelector('.container') || document.body;
  container.insertBefore(errorDiv, container.firstChild);

  setTimeout(() => errorDiv.remove(), 5000);
}

function clearFileSelection() {
  selectedFile = null;
  document.getElementById('fileInput').value = '';
  document.getElementById('fileName').textContent = '';
  document.getElementById('original').src = '';
}

document.addEventListener('DOMContentLoaded', function() {
  const customFileLabel = document.querySelector('.custom-file');
  const fileInput = document.getElementById('fileInput');

  if (customFileLabel) {
    customFileLabel.addEventListener('click', function() {
      fileInput.click();
    });
  }
});

document.addEventListener('DOMContentLoaded', function() {
  const imageContainer = document.querySelector('.image-container');

  if (imageContainer) {
    imageContainer.addEventListener('dragover', (e) => {
      e.preventDefault();
      imageContainer.style.borderColor = '#3B82F6';
    });

    imageContainer.addEventListener('dragleave', () => {
      imageContainer.style.borderColor = 'rgba(59, 130, 246, 0.3)';
    });

    imageContainer.addEventListener('drop', (e) => {
      e.preventDefault();
      imageContainer.style.borderColor = 'rgba(59, 130, 246, 0.3)';

      const files = e.dataTransfer.files;
      if (files.length > 0) {
        document.getElementById('fileInput').files = files;
        const event = new Event('change', { bubbles: true });
        document.getElementById('fileInput').dispatchEvent(event);
      }
    });
  }
});
