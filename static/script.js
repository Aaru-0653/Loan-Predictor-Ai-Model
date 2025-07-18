// filepath: loan-prediction-form/loan-prediction-form/src/script.js

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('loanForm');
    const inputs = form.querySelectorAll('input, select');
    
    // Add hover effect to inputs
    inputs.forEach(input => {
        input.addEventListener('focus', () => {
            input.classList.add('hover');
        });
        input.addEventListener('blur', () => {
            input.classList.remove('hover');
        });
    });

    // Form submission event
    form.addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent default form submission

        // Perform validation (example: check if all fields are filled)
        let isValid = true;
        inputs.forEach(input => {
            if (!input.value) {
                isValid = false;
                input.classList.add('error');
            } else {
                input.classList.remove('error');
            }
        });

        if (isValid) {
            // Here you can handle the form submission, e.g., send data to the server
            alert('Form submitted successfully!');
            form.reset(); // Reset the form after submission
        } else {
            alert('Please fill in all fields.');
        }
    });
});