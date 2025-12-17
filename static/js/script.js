$(document).ready(function() {
    // Add animations to elements when they come into view
    const animateElements = $('.animate__animated');
    
    $(window).scroll(function() {
        animateElements.each(function() {
            const elementPos = $(this).offset().top;
            const topOfWindow = $(window).scrollTop();
            
            if (elementPos < topOfWindow + $(window).height() - 100) {
                $(this).addClass('animate__fadeInUp');
            }
        });
    });
    
    // Form validation
    $('form').submit(function(e) {
        let isValid = true;
        
        $(this).find('input[required]').each(function() {
            if (!$(this).val()) {
                isValid = false;
                $(this).addClass('error');
            } else {
                $(this).removeClass('error');
            }
        });
        
        if (!isValid) {
            e.preventDefault();
            alert('Please fill in all required fields');
        }
    });
    
    // Password confirmation validation
    $('#confirm-password').keyup(function() {
        const password = $('#password').val();
        const confirmPassword = $(this).val();
        
        if (password !== confirmPassword) {
            $(this).addClass('error');
        } else {
            $(this).removeClass('error');
        }
    });
});