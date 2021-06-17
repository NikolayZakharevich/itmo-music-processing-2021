<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="content-type" content="text/html;charset=UTF-8"/>
    <meta name='viewport' content='initial-scale=1,maximum-scale=1,user-scalable=no'/>

    <title>Deflamel &#8211; Design Wizard</title>
    <script src="https://ajax.googleapis.com/ajax/libs/webfont/1.6.26/webfont.js"></script>
    <link rel='stylesheet' id='vapp-responsive-css' href='https://app.deflamel.com/assets/styles/all.css'
          type='text/css' media='all'/>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.2.1/css/bootstrap.min.css"
          integrity="sha384-GJzZqFGwb1QTTN6wy59ffF1BuGJpLSa9DkKMp0DgiMDm4iYMj70gZWKYbI706tWS" crossorigin="anonymous"/>

    <script src='https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js'></script>
    <script src='http://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js'></script>
    <style>
        .masonry {
            column-count: 1;
            column-gap: 1em;
            margin-top: 50px;
        }

        .masonry-item {
            display: inline-block;
            border: 1px solid #ddd;
            box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.18);
            border-radius: .5rem;
            padding: 1.5rem;
            margin: .5rem auto;
            width: 100%;
            text-align: center;
        }
    </style>
</head>
<body>
<section>
    <article class="SignUp">
        <header class="Header">
            <div class="Header__BackButton">
                <a href="https://deflamel.com/index.php/design-features/" class="Header__BackButton-Arrow"></a>
            </div>
            <a class="Header__Logo" href="//deflamel.com/" target="_self"></a>
            <a href="https://deflamel.com/index.php/design-features/">
                <div class="Header__Menu">
                    <span class="Header__Menu-Item"></span>
                    <span class="Header__Menu-Item"></span>
                    <span class="Header__Menu-Item"></span>
                </div>
            </a>
        </header>

        <div class="Center" style="padding-top: 150px; justify-content: flex-start;">

            <h1 class="Title">Music Keywords Suggestion</h1>
            <p class="SubTitle">Get design ideas based on the song</p>

            <#if keywords??>
            <div class="masonry col-12">
                <h4 class="Title">Your song's keywords:</h4>
                <div class="masonry-item">
                    <p style="font-family: ${font}; font-size: 1.5rem;">${keywords_str}</p>
                </div>
            </div>
            <#else></#if>

            <form method='POST' enctype='multipart/form-data' action="/demo/music-keywords?step=1">
                <div class="form-group">
                    <div style="clear: both; margin: 20px 0 50px 0;">
                        <label for="audio">Upload the song</label>
                        <input type="file" id="audio" name="audio"
                               accept="audio/*" class="Input__Input" required>
                    </div>
                </div>
                <input type='submit'
                       class="Footer__NextButton Footer__NextButton--visible Footer__NextButton--active"
                       value='Submit'/>
            </form>
        </div>
    </article>
</section>
</body>
</html>