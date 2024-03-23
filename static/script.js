document.getElementById('prediction-form').addEventListener('submit', function(event) {
    event.preventDefault(); 
    
    var formData = {
        'ph': parseFloat(document.getElementById('ph').value),
        'hardness': parseFloat(document.getElementById('hardness').value)
        
    };

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 'features': formData })
    })
    .then(response => response.json())
    .then(data => {
        var resultContainer = document.getElementById('result-container');
        if (data.water_quality === 'Good') {
            resultContainer.innerHTML = '<p>Water is good for plants.</p>';
        } else {
            resultContainer.innerHTML = '<p>Water is not good for plants.</p>';
        }
    })
    .catch(error => console.error('Error:', error));
});
