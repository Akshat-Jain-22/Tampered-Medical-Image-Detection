const API_BASE = "/api";
let selectedFile = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
  const fileInput = document.getElementById('fileInput');
  fileInput.addEventListener('change', handleFileSelect);
});

// Handle file selection
function handleFileSelect(event) {
  const file = event.target.files[0];
  if (!file) return;

  // Check file type
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

  // Display filename
  document.getElementById('fileName').textContent = `Selected: ${file.name}`;

  // Display file preview (for image files only)
  const reader = new FileReader();
  reader.onload = function(e) {
    // Check if it's DICOM (binary format)
    if (!file.name.toLowerCase().endsWith('.dcm')) {
      document.getElementById('original').src = e.target.result;
      document.getElementById('original').style.display = 'block';
    } else {
      document.getElementById('original').innerHTML = '<p>DICOM file selected - analyzing will show preview</p>';
    }
  };
  reader.readAsDataURL(file);
}

// Analyze image
async function analyzeImage() {
  if (!selectedFile) {
    showError('Please select an image file first');
    return;
  }

  const formData = new FormData();
  formData.append('file', selectedFile);

  // Show loading
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

// Display results
function displayResults(result) {
  // Classification
  const classification = result.classification;
  const statusElement = document.getElementById('status');
  statusElement.textContent = classification;
  statusElement.className = 'status-text ' + classification.toLowerCase();

  // Probabilities
  const tamperedProb = (result.tampered_probability * 100).toFixed(1);
  const authenticProb = (result.authentic_probability * 100).toFixed(1);

  document.getElementById('tamperedProb').textContent = tamperedProb;
  document.getElementById('authenticProb').textContent = authenticProb;

  // Confidence
  const confidence = classification === 'Tampered' ? tamperedProb : authenticProb;
  document.getElementById('confidence').innerHTML = `<strong>Confidence:</strong> ${confidence}%`;

  // Risk gauge
  const riskLevel = (result.tampered_probability * 100).toFixed(0);
  updateGauge(riskLevel);

  // Heatmap - convert array back to image
  if (result.heatmap) {
    displayHeatmap(result.heatmap);
  } else if (result.heatmap_path) {
    document.getElementById('heatmapImage').src = result.heatmap_path;
  }

  // Show results section
  document.getElementById('results').style.display = 'block';
  document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

// Display heatmap
function displayHeatmap(heatmapArray) {
  // Create canvas and draw heatmap
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  // Assume 224x224 heatmap (from model preprocessing)
  const width = 224;
  const height = 224;
  canvas.width = width;
  canvas.height = height;

  // Create image data from array
  const imageData = ctx.createImageData(width, height);
  const data = imageData.data;

  let dataIndex = 0;
  for (let i = 0; i < heatmapArray.length; i++) {
    const pixelArray = heatmapArray[i];
    for (let j = 0; j < pixelArray.length; j++) {
      const pixelValues = pixelArray[j];
      data[dataIndex] = Math.round(pixelValues[0] * 255);      // R
      data[dataIndex + 1] = Math.round(pixelValues[1] * 255);  // G
      data[dataIndex + 2] = Math.round(pixelValues[2] * 255);  // B
      data[dataIndex + 3] = 255;                               // A
      dataIndex += 4;
    }
  }

  ctx.putImageData(imageData, 0, 0);

  // Convert to image
  const heatmapImage = document.getElementById('heatmapImage');
  heatmapImage.src = canvas.toDataURL();
}

// Update risk gauge
function updateGauge(percent) {
  percent = Math.min(Math.max(percent, 0), 100);

  const fill = document.getElementById('gaugeFill');
  const text = document.getElementById('gaugeText');

  // Calculate rotation: -90deg to +90deg (180deg total)
  const rotation = (percent / 100) * 180 - 90;
  fill.style.transform = `rotate(${rotation}deg)`;

  text.textContent = percent + '%';

  // Change color based on risk
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

// Show loading state
function showLoading(show) {
  document.getElementById('loading').style.display = show ? 'block' : 'none';
  const analyzeBtn = document.querySelector('button[onclick="analyzeImage()"]');
  if (analyzeBtn) {
    analyzeBtn.disabled = show;
  }
}

// Show results section
function hideResults() {
  document.getElementById('results').style.display = 'none';
}

// Show error
function showError(message) {
  // Clear previous error
  const existingError = document.querySelector('.error');
  if (existingError) {
    existingError.remove();
  }

  const errorDiv = document.createElement('div');
  errorDiv.className = 'error';
  errorDiv.textContent = '❌ ' + message;

  const container = document.querySelector('.container') || document.body;
  container.insertBefore(errorDiv, container.firstChild);

  // Remove after 5 seconds
  setTimeout(() => errorDiv.remove(), 5000);
}

// Clear file selection
function clearFileSelection() {
  selectedFile = null;
  document.getElementById('fileInput').value = '';
  document.getElementById('fileName').textContent = '';
  document.getElementById('original').src = '';
}

// Handle file input click
document.addEventListener('DOMContentLoaded', function() {
  const customFileLabel = document.querySelector('.custom-file');
  const fileInput = document.getElementById('fileInput');

  if (customFileLabel) {
    customFileLabel.addEventListener('click', function() {
      fileInput.click();
    });
  }
});

// Drag and drop support
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
