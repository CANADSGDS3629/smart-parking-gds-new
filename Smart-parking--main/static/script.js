document.addEventListener("DOMContentLoaded", function() {
    const startButton = document.getElementById("startButton");
    startButton.addEventListener("click", function() {
        // Send request to Flask backend to start processing
        fetch('/process', {method: 'POST'})
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.text();
            })
            .then(data => console.log(data))
            .catch(error => console.error('Error:', error));
    });
});
