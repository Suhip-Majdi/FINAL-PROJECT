// Start Webcam feed on button click
function startWebcam() {
    document.getElementById('video-container').style.display = 'block';
    document.getElementById('webcam-feed').src = "/video_feed"; // Start the webcam feed
}

function stopWebcam() {
    document.getElementById('video-container').style.display = 'none';
    document.getElementById('webcam-feed').src = ""; // Stop loading the webcam feed

    // Send a request to the server to stop processing
    fetch('/stop_webcam', { method: 'POST' });
}

// Upload Image Handling
document.getElementById('upload-form').addEventListener('submit', function (event) {
    event.preventDefault();  // Prevent default form submission

    const formData = new FormData();
    const fileInput = document.getElementById('file-upload');  // Corrected the ID here

    // Check if a file is selected
    if (fileInput.files.length === 0) {
        alert('Please select an image file.');
        return;
    }

    formData.append('file', fileInput.files[0]);

    // Send the form data with the image file
    fetch('/upload_image', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        const processedImage = data.image;
        const detectionInfo = data.detection_info;

        // Display the processed image
        document.getElementById('processed-image').src = `data:image/jpeg;base64,${processedImage}`;
        document.getElementById('processed-image').style.display = 'block';  // Show the image

        // Show the custom modal with buttons for DIY and Amazon links
        showAlert(detectionInfo);
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while uploading the image.');
    });
});

// Function to show the custom modal with DIY and Price buttons
function showAlert(detectionInfo) {
    const objectName = detectionInfo.object_name;
    const diyUrl = detectionInfo.diy_url;
    const priceUrl = detectionInfo.price_url;

    // Determine the message based on the detected object
    let objectDescription;
    if (objectName === 'Spark-Plugs') {
        objectDescription = 'Electrical device used in an internal combustion engine to produce a spark which ignites the air-fuel mixture in the combustion chamber.';
    } else if (objectName === 'Battery') {
        objectDescription = 'A rechargeable battery that is used to start a motor vehicle.';
    } else if (objectName === 'Shock-Absorber') {
        objectDescription = 'Designed to absorb or dampen the compression and rebound of the springs and suspension.';
    } else {
        objectDescription = 'No Object Detected.';
    }

    // Construct the alert message
    const alertMessage = `Detected Object: ${objectName}\nDescription: ${objectDescription}\nWhat would you like to do?`;

    // Display the custom modal and set the message
    document.getElementById('modal-message').innerText = alertMessage;
    document.getElementById('custom-modal').style.display = 'block';

    // Remove any existing event listeners before adding new ones
    const diyBtn = document.getElementById('diy-btn');
    const priceBtn = document.getElementById('price-btn');

    // Remove old listeners by re-cloning the button to reset listeners
    diyBtn.replaceWith(diyBtn.cloneNode(true));
    priceBtn.replaceWith(priceBtn.cloneNode(true));

    // Handle DIY button click
    document.getElementById('diy-btn').addEventListener('click', function () {
        window.open(diyUrl, '_blank'); // Open DIY URL (YouTube)
        document.getElementById('custom-modal').style.display = 'none'; // Hide modal after action
    });

    // Handle Price button click
    document.getElementById('price-btn').addEventListener('click', function () {
        window.open(priceUrl, '_blank'); // Open Price URL (Amazon)
        document.getElementById('custom-modal').style.display = 'none'; // Hide modal after action
    });
}
