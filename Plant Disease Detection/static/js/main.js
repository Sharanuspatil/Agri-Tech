$(document).ready(function () {

    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();


    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });


    $('#btn-predict').click(function () {
      
        var form_data = new FormData($('#upload-file')[0]);

        $(this).hide();
        $('.loader').show();
        

       
        $.ajax({
            type: 'POST',
            url: '/predict',
            
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                
                $('.loader').hide();
                $('#result').fadeIn(600);
                
                console.log('Success!');
            },
        });
    });

});