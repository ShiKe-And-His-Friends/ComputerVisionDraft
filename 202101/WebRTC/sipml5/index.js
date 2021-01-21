<!DOCTYPE html>
<!--
* Copyright (C) 2012-2018 Doubango Telecom <http://www.doubango.org>
* License: BSD
* This file is part of Open Source sipML5 solution <http://www.sipml5.org>
-->
<html>
<!-- head -->
<head>
    <meta charset="utf-8" />
    <title>sipML5 live demo</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="Keywords" content="doubango, sipML5, VoIP, HTML5, WebRTC, RTCWeb, SIP, IMS, Video chat, VP8, live demo " />
    <meta name="Description" content="HTML5 SIP client using WebRTC framework" />
    <meta name="author" content="Doubango Telecom" />

    <!-- SIPML5 API:
    DEBUG VERSION: 'SIPml-api.js'
    RELEASE VERSION: 'release/SIPml-api.js'
    -->
    <!-- <script src="SIPml-api.js?svn=252" type="text/javascript"> </script> -->
    <script src="idcvideo-sdk.js" type="text/javascript"> </script>

    <!-- Styles -->
    <link href="./assets/css/bootstrap.css" rel="stylesheet" />
    <style type="text/css">
        body {
            padding-top: 80px;
            padding-bottom: 40px;
        }

        .navbar-inner-red {
            background-color: #600000;
            background-image: none;
            background-repeat: no-repeat;
            filter: none;
        }

        .full-screen {
            position: absolute;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }

        .normal-screen {
            position: relative;
        }

        .call-options {
            padding: 5px;
            background-color: #f0f0f0;
            border: 1px solid #eee;
            border: 1px solid rgba(0, 0, 0, 0.08);
            -webkit-border-radius: 4px;
            -moz-border-radius: 4px;
            border-radius: 4px;
            -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
            -moz-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
            box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
            -webkit-transition-property: opacity;
            -moz-transition-property: opacity;
            -o-transition-property: opacity;
            -webkit-transition-duration: 2s;
            -moz-transition-duration: 2s;
            -o-transition-duration: 2s;
        }

        .tab-video,
        .div-video {
            width: 100%;
            height: 0px;
            -webkit-transition-property: height;
            -moz-transition-property: height;
            -o-transition-property: height;
            -webkit-transition-duration: 2s;
            -moz-transition-duration: 2s;
            -o-transition-duration: 2s;
        }

        .label-align {
            display: block;
            padding-left: 15px;
            text-indent: -15px;
        }

        .input-align {
            width: 13px;
            height: 13px;
            padding: 0;
            margin: 0;
            vertical-align: bottom;
            position: relative;
            top: -1px;
            *overflow: hidden;
        }

        .glass-panel {
            z-index: 99;
            position: fixed;
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 0;
            top: 0;
            left: 0;
            opacity: 0.8;
            background-color: Gray;
        }

        .div-keypad {
            z-index: 100;
            position: fixed;
            -moz-transition-property: left top;
            -o-transition-property: left top;
            -webkit-transition-duration: 2s;
            -moz-transition-duration: 2s;
            -o-transition-duration: 2s;
        }

        .previewvideo {
            position: absolute;
            width: 88px;
            height: 72px;
            margin-top: -42px;
        }
    </style>
    <link href="./assets/css/bootstrap-responsive.css" rel="stylesheet" />
    <!-- Le fav and touch icons -->
    <link rel="shortcut icon" href="./assets/ico/favicon.ico" />
    <link rel="apple-touch-icon-precomposed" sizes="114x114" href="./assets/ico/apple-touch-icon-114-precomposed.png" />
    <link rel="apple-touch-icon-precomposed" sizes="72x72" href="./assets/ico/apple-touch-icon-72-precomposed.png" />
    <link rel="apple-touch-icon-precomposed" href="./assets/ico/apple-touch-icon-57-precomposed.png" />

    <!-- Javascript code -->
    <script type="text/javascript">

        var cbAVPFDisable;
        var txtWebsocketServerUrl;
        var txtSIPOutboundProxyUrl;
        var txtInfo;

        var sTransferNumber;
        var oRingTone, oRingbackTone;
        var oSipStack, oSipSessionRegister, oSipSessionCall, oSipSessionTransferCall;
        var videoRemote, videoLocal, audioRemote;
        var bFullScreen = false;
        var oNotifICall;
        var bDisableVideo = false;
        var viewVideoLocal, viewVideoRemote, viewLocalScreencast;
        var oConfigCall;
        var oReadyStateTimer;

        C =
        {
            divKeyPadWidth: 220
        };

        window.onload = function () {
            cbRTCWebBreaker = document.getElementById("cbRTCWebBreaker");
            txtWebsocketServerUrl = document.getElementById("txtWebsocketServerUrl");
            txtSIPOutboundProxyUrl = document.getElementById("txtSIPOutboundProxyUrl");
            txtInfo = document.getElementById("txtInfo");

            //txtWebsocketServerUrl.disabled = !window.WebSocket || navigator.appName == "Microsoft Internet Explorer"; // Do not use WS on IE
            document.getElementById("btnSave").disabled = !window.localStorage;
            document.getElementById("btnRevert").disabled = !window.localStorage;

            if(window.localStorage){
                settingsRevert(true);
            }

            // window.console && window.console.info && window.console.info("location=" + window.location);

            videoLocal = document.getElementById("video_local");
            videoRemote = document.getElementById("video_remote");
            audioRemote = document.getElementById("audio_remote");

            divCallCtrl.onmousemove = onDivCallCtrlMouseMove;

            // set debug level
            // SIPml.setDebugLevel((window.localStorage && window.localStorage.getItem('org.doubango.expert.disable_debug') == "true") ? "error" : "info");

            loadCredentials();
            loadCallOptions();

            // Initialize call button
            uiBtnCallSetText("Call");

            var getPVal = function (PName) {
                var query = window.location.search.substring(1);
                var vars = query.split('&');
                for (var i = 0; i < vars.length; i++) {
                    var pair = vars[i].split('=');
                    if (decodeURIComponent(pair[0]) === PName) {
                        return decodeURIComponent(pair[1]);
                    }
                }
                return null;
            }

            var preInit = function () {
                // initialize SIPML5
                idcMediaInit({
                    error:(error)=> {
                        alert(error);
                    },
                    success:postInit
                });
            }

            oReadyStateTimer = setInterval(function () {
                if (document.readyState === "complete") {
                    clearInterval(oReadyStateTimer);
                    preInit();
                }
            },
            500);
        };

        function settingsSave() {
            window.localStorage.setItem('org.doubango.expert.enable_rtcweb_breaker', cbRTCWebBreaker.checked ? "true" : "false");
            if (!txtWebsocketServerUrl.disabled) {
                window.localStorage.setItem('org.doubango.expert.websocket_server_url', txtWebsocketServerUrl.value);
            }
            window.localStorage.setItem('org.doubango.expert.sip_outboundproxy_url', txtSIPOutboundProxyUrl.value);
            window.localStorage.setItem('org.doubango.expert.ice_servers', txtIceServers.value);
            window.localStorage.setItem('org.doubango.expert.bandwidth', txtBandwidth.value);
            window.localStorage.setItem('org.doubango.expert.video_size', txtSizeVideo.value);

            txtInfo.innerHTML = '<i>Saved</i>';
        }

        function settingsRevert(bNotUserAction) {
            cbRTCWebBreaker.checked = true;//(window.localStorage.getItem('org.doubango.expert.enable_rtcweb_breaker') == "true");
            txtWebsocketServerUrl.value = (window.localStorage.getItem('org.doubango.expert.websocket_server_url') || "");
            if (!txtWebsocketServerUrl.value || txtWebsocketServerUrl.value == "") {
                txtWebsocketServerUrl.value = "wss://stdcall.freevoip.com.cn:40001/ws";
            }
            txtSIPOutboundProxyUrl.value = (window.localStorage.getItem('org.doubango.expert.sip_outboundproxy_url') || "");
            txtIceServers.value = (window.localStorage.getItem('org.doubango.expert.ice_servers') || "");
            if (!txtIceServers.value || txtIceServers.value == "") {
                txtIceServers.value = "[]";
            }
            txtBandwidth.value = (window.localStorage.getItem('org.doubango.expert.bandwidth') || "");
            txtSizeVideo.value = (window.localStorage.getItem('org.doubango.expert.video_size') || "");


            if (!bNotUserAction) {
                txtInfo.innerHTML = '<i>Reverted</i>';
            }
        }

        function postInit() {
            
            // FIXME: displays must be per session
            viewVideoLocal = videoLocal;
            viewVideoRemote = videoRemote;

            btnRegister.disabled = false;
            document.body.style.cursor = 'default';
            oConfigCall = {
                audio_remote: audioRemote,
                video_local: viewVideoLocal,
                video_remote: viewVideoRemote,
                screencast_window_id: 0x00000000, // entire desktop
                bandwidth: { audio: undefined, video: undefined },
                video_size: { minWidth: undefined, minHeight: undefined, maxWidth: undefined, maxHeight: undefined },
                events_listener: { events: '*', listener: onSipEventSession },
                sip_caps: [
                                { name: '+g.oma.sip-im' },
                                { name: 'language', value: '\"en,fr\"' }
                ]
            };
        }


        function loadCallOptions() {
            if (window.localStorage) {
                var s_value;
                if ((s_value = window.localStorage.getItem('org.doubango.call.phone_number'))) txtPhoneNumber.value = s_value;
                bDisableVideo = (window.localStorage.getItem('org.doubango.expert.disable_video') == "true");

                txtCallStatus.innerHTML = '<i>Video ' + (bDisableVideo ? 'disabled' : 'enabled') + '</i>';
            }
        }

        function saveCallOptions() {
            if (window.localStorage) {
                window.localStorage.setItem('org.doubango.call.phone_number', txtPhoneNumber.value);
                window.localStorage.setItem('org.doubango.expert.disable_video', bDisableVideo ? "true" : "false");
            }
        }

        function loadCredentials() {
            if (window.localStorage) {
                // IE retuns 'null' if not defined
                var s_value;
                if ((s_value = window.localStorage.getItem('org.doubango.identity.display_name'))) txtDisplayName.value = s_value;
                if ((s_value = window.localStorage.getItem('org.doubango.identity.impi'))) txtPrivateIdentity.value = s_value;
                if ((s_value = window.localStorage.getItem('org.doubango.identity.impu'))) txtPublicIdentity.value = s_value;
                if ((s_value = window.localStorage.getItem('org.doubango.identity.password'))) txtPassword.value = s_value;
                if ((s_value = window.localStorage.getItem('org.doubango.identity.realm'))) txtRealm.value = s_value;

                if (!txtDisplayName.value || txtDisplayName.value == "") {
                    txtDisplayName.value = "601";
                }

                if (!txtPrivateIdentity.value || txtPrivateIdentity.value == "") {
                    txtPrivateIdentity.value = "601";
                }

                if (!txtPublicIdentity.value || txtPublicIdentity.value == "") {
                    txtPublicIdentity.value = "sip:601@stdcall.freevoip.com.cn";
                }

                if (!txtRealm.value || txtRealm.value == "") {
                    txtRealm.value = "stdcall.freevoip.com.cn";
                }
            }
        };

        function saveCredentials() {
            if (window.localStorage) {
                window.localStorage.setItem('org.doubango.identity.display_name', txtDisplayName.value);
                window.localStorage.setItem('org.doubango.identity.impi', txtPrivateIdentity.value);
                window.localStorage.setItem('org.doubango.identity.impu', txtPublicIdentity.value);
                window.localStorage.setItem('org.doubango.identity.password', txtPassword.value);
                window.localStorage.setItem('org.doubango.identity.realm', txtRealm.value);
            }
        };

        // sends SIP REGISTER request to login
        function sipRegister() {
            // catch exception for IE (DOM not ready)
            try {
                btnRegister.disabled = true;
                if (!txtRealm.value || !txtPrivateIdentity.value || !txtPublicIdentity.value) {
                    txtRegStatus.innerHTML = '<b>Please fill madatory fields (*)</b>';
                    btnRegister.disabled = false;
                    return;
                }
                var o_impu = tsip_uri.prototype.Parse(txtPublicIdentity.value);
                if (!o_impu || !o_impu.s_user_name || !o_impu.s_host) {
                    txtRegStatus.innerHTML = "<b>[" + txtPublicIdentity.value + "] is not a valid Public identity</b>";
                    btnRegister.disabled = false;
                    return;
                }

                // enable notifications if not already done
                if (window.webkitNotifications && window.webkitNotifications.checkPermission() != 0) {
                    window.webkitNotifications.requestPermission();
                }

                // save credentials
                saveCredentials();


                // create SIP stack
                oSipStack = new SIPml.Stack({
                    realm: txtRealm.value,
                    impi: txtPrivateIdentity.value,
                    impu: txtPublicIdentity.value,
                    password: txtPassword.value,
                    display_name: txtDisplayName.value,
                    websocket_proxy_url: (window.localStorage ? window.localStorage.getItem('org.doubango.expert.websocket_server_url') : null),
                    outbound_proxy_url: (window.localStorage ? window.localStorage.getItem('org.doubango.expert.sip_outboundproxy_url') : null),
                    ice_servers: (window.localStorage ? window.localStorage.getItem('org.doubango.expert.ice_servers') : null),
                    enable_rtcweb_breaker: (window.localStorage ? window.localStorage.getItem('org.doubango.expert.enable_rtcweb_breaker') == "true" : false),
                    events_listener: { events: '*', listener: onSipEventStack },
                    enable_early_ims: (window.localStorage ? window.localStorage.getItem('org.doubango.expert.disable_early_ims') != "true" : true), // Must be true unless you're using a real IMS network
                    enable_media_stream_cache: (window.localStorage ? window.localStorage.getItem('org.doubango.expert.enable_media_caching') == "true" : false),
                    bandwidth: (window.localStorage ? tsk_string_to_object(window.localStorage.getItem('org.doubango.expert.bandwidth')) : null), // could be redefined a session-level
                    video_size: (window.localStorage ? tsk_string_to_object(window.localStorage.getItem('org.doubango.expert.video_size')) : null), // could be redefined a session-level
                    sip_headers: [
                            { name: 'User-Agent', value: 'IM-client/OMA1.0 sipML5-v1.2016.03.04' },
                            { name: 'Organization', value: 'Doubango Telecom' }
                    ]
                }
                );
                if (oSipStack.start() != 0) {
                    txtRegStatus.innerHTML = '<b>Failed to start the SIP stack</b>';
                }
                else return;
            }
            catch (e) {
                txtRegStatus.innerHTML = "<b>2:" + e + "</b>";
            }
            btnRegister.disabled = false;
        }

        // sends SIP REGISTER (expires=0) to logout
        function sipUnRegister() {
            if (oSipStack) {
                oSipStack.stop(); // shutdown all sessions
            }
        }

        // makes a call (SIP INVITE)
        function sipCall(s_type) {
            if (oSipStack && !oSipSessionCall && !tsk_string_is_null_or_empty(txtPhoneNumber.value)) {
                if (s_type == 'call-screenshare') {
                    if (!SIPml.isScreenShareSupported()) {
                        alert('Screen sharing not supported. Are you using chrome 26+?');
                        return;
                    }
                    if (!location.protocol.match('https')) {
                        if (confirm("Screen sharing requires https://. Do you want to be redirected?")) {
                            sipUnRegister();
                            window.location = 'https://ns313841.ovh.net/call.htm';
                        }
                        return;
                    }
                }
                btnCall.disabled = true;
                btnHangUp.disabled = false;

                if (window.localStorage) {
                    oConfigCall.bandwidth = tsk_string_to_object(window.localStorage.getItem('org.doubango.expert.bandwidth')); // already defined at stack-level but redifined to use latest values
                    oConfigCall.video_size = tsk_string_to_object(window.localStorage.getItem('org.doubango.expert.video_size')); // already defined at stack-level but redifined to use latest values
                }

                // create call session
                oSipSessionCall = oSipStack.newSession(s_type, oConfigCall);
                // make call
                if (oSipSessionCall.call(txtPhoneNumber.value) != 0) {
                    oSipSessionCall = null;
                    txtCallStatus.value = 'Failed to make call';
                    btnCall.disabled = false;
                    btnHangUp.disabled = true;
                    return;
                }
                saveCallOptions();
            }
            else if (oSipSessionCall) {
                txtCallStatus.innerHTML = '<i>Connecting...</i>';
                oSipSessionCall.accept(oConfigCall);
            }
        }

        // Share entire desktop aor application using BFCP or WebRTC native implementation
        function sipShareScreen() {
            
        }

        // Mute or Unmute the call
        function sipToggleMute() {
            if (oSipSessionCall) {
                var i_ret;
                var bMute = !oSipSessionCall.bMute;
                txtCallStatus.innerHTML = bMute ? '<i>Mute the call...</i>' : '<i>Unmute the call...</i>';
                i_ret = oSipSessionCall.mute('audio'/*could be 'video'*/, bMute);
                if (i_ret != 0) {
                    txtCallStatus.innerHTML = '<i>Mute / Unmute failed</i>';
                    return;
                }
                oSipSessionCall.bMute = bMute;
                btnMute.value = bMute ? "Unmute" : "Mute";
            }
        }

        // terminates the call (SIP BYE or CANCEL)
        function sipHangUp() {
            if (oSipSessionCall) {
                txtCallStatus.innerHTML = '<i>Terminating the call...</i>';
                oSipSessionCall.hangup({ events_listener: { events: '*', listener: onSipEventSession } });
            }
        }

        function sipSendDTMF(c) {
            if (oSipSessionCall && c) {
                if (oSipSessionCall.dtmf(c) == 0) {
                    try { dtmfTone.play(); } catch (e) { }
                }
            }
        }

        function startRingTone() {
            try { ringtone.play(); }
            catch (e) { }
        }

        function stopRingTone() {
            try { ringtone.pause(); }
            catch (e) { }
        }

        function startRingbackTone() {
            try { ringbacktone.play(); }
            catch (e) { }
        }

        function stopRingbackTone() {
            try { ringbacktone.pause(); }
            catch (e) { }
        }

        function showNotifICall(s_number) {
            // permission already asked when we registered
            if (window.webkitNotifications && window.webkitNotifications.checkPermission() == 0) {
                if (oNotifICall) {
                    oNotifICall.cancel();
                }
                oNotifICall = window.webkitNotifications.createNotification('images/sipml-34x39.png', 'Incaming call', 'Incoming call from ' + s_number);
                oNotifICall.onclose = function () { oNotifICall = null; };
                oNotifICall.show();
            }
        }

        function onDivCallCtrlMouseMove(evt) {
            try { // IE: DOM not ready
                if (tsk_utils_have_stream()) {
                    btnCall.disabled = (!tsk_utils_have_stream() || !oSipSessionRegister || !oSipSessionRegister.is_connected());
                    document.getElementById("divCallCtrl").onmousemove = null; // unsubscribe
                }
            }
            catch (e) { }
        }

        function uiOnConnectionEvent(b_connected, b_connecting) { // should be enum: connecting, connected, terminating, terminated
            btnRegister.disabled = b_connected || b_connecting;
            btnUnRegister.disabled = !b_connected && !b_connecting;
            btnCall.disabled = !(b_connected && tsk_utils_have_webrtc() && tsk_utils_have_stream());
            btnHangUp.disabled = !oSipSessionCall;
        }

        function uiVideoDisplayEvent(b_local, b_added) {
            var o_elt_video = b_local ? videoLocal : videoRemote;

            if (b_added) {
                o_elt_video.style.opacity = 1;
                uiVideoDisplayShowHide(true);
            }
            else {
                o_elt_video.style.opacity = 0;
            }
        }

        function uiVideoDisplayShowHide(b_show) {
            if (b_show) {
                tdVideo.style.height = '340px';
                divVideo.style.height = navigator.appName == 'Microsoft Internet Explorer' ? '100%' : '340px';
            }
            else {
                tdVideo.style.height = '0px';
                divVideo.style.height = '0px';
            }
        }

        function uiDisableCallOptions() {
            if (window.localStorage) {
                window.localStorage.setItem('org.doubango.expert.disable_callbtn_options', 'true');
                uiBtnCallSetText('Call');
                alert('Use expert view to enable the options again (/!\\requires re-loading the page)');
            }
        }

        function uiBtnCallSetText(s_text) {
            switch (s_text) {
                case "Call":
                    {
                        var bDisableCallBtnOptions = true;//(window.localStorage && window.localStorage.getItem('org.doubango.expert.disable_callbtn_options') == "true");
                        btnCall.value = btnCall.innerHTML = bDisableCallBtnOptions ? 'Call' : 'Call <span id="spanCaret" class="caret">';
                        btnCall.setAttribute("class", bDisableCallBtnOptions ? "btn btn-primary" : "btn btn-primary dropdown-toggle");
                        btnCall.onclick = bDisableCallBtnOptions ? function () { sipCall(bDisableVideo ? 'call-audio' : 'call-audiovideo'); } : null;
                        ulCallOptions.style.visibility = bDisableCallBtnOptions ? "hidden" : "visible";
                        if (!bDisableCallBtnOptions && ulCallOptions.parentNode != divBtnCallGroup) {
                            divBtnCallGroup.appendChild(ulCallOptions);
                        }
                        else if (bDisableCallBtnOptions && ulCallOptions.parentNode == divBtnCallGroup) {
                            document.body.appendChild(ulCallOptions);
                        }

                        break;
                    }
                default:
                    {
                        btnCall.value = btnCall.innerHTML = s_text;
                        btnCall.setAttribute("class", "btn btn-primary");
                        btnCall.onclick = function () { sipCall(bDisableVideo ? 'call-audio' : 'call-audiovideo'); };
                        ulCallOptions.style.visibility = "hidden";
                        if (ulCallOptions.parentNode == divBtnCallGroup) {
                            document.body.appendChild(ulCallOptions);
                        }
                        break;
                    }
            }
        }

        function uiCallTerminated(s_description) {
            uiBtnCallSetText("Call");
            btnHangUp.value = 'HangUp';
            btnMute.value = "Mute";
            btnCall.disabled = false;
            btnHangUp.disabled = true;
            if (window.btnBFCP) window.btnBFCP.disabled = true;

            oSipSessionCall = null;

            stopRingbackTone();
            stopRingTone();

            txtCallStatus.innerHTML = "<i>" + s_description + "</i>";
            uiVideoDisplayShowHide(false);
            divCallOptions.style.opacity = 0;

            if (oNotifICall) {
                oNotifICall.cancel();
                oNotifICall = null;
            }

            uiVideoDisplayEvent(false, false);
            uiVideoDisplayEvent(true, false);

            setTimeout(function () { if (!oSipSessionCall) txtCallStatus.innerHTML = ''; }, 2500);
        }

        // Callback function for SIP Stacks
        function onSipEventStack(e /*SIPml.Stack.Event*/) {
            tsk_utils_log_info('==stack event = ' + e.type);
            switch (e.type) {
                case 'started':
                    {
                        // catch exception for IE (DOM not ready)
                        try {
                            // LogIn (REGISTER) as soon as the stack finish starting
                            oSipSessionRegister = this.newSession('register', {
                                expires: 200,
                                events_listener: { events: '*', listener: onSipEventSession },
                                sip_caps: [
                                            { name: '+g.oma.sip-im', value: null },
                                            //{ name: '+sip.ice' }, // rfc5768: FIXME doesn't work with Polycom TelePresence
                                            { name: '+audio', value: null },
                                            { name: 'language', value: '\"en,fr\"' }
                                ]
                            });
                            oSipSessionRegister.register();
                        }
                        catch (e) {
                            txtRegStatus.value = txtRegStatus.innerHTML = "<b>1:" + e + "</b>";
                            btnRegister.disabled = false;
                        }
                        break;
                    }
                case 'stopping': case 'stopped': case 'failed_to_start': case 'failed_to_stop':
                    {
                        var bFailure = (e.type == 'failed_to_start') || (e.type == 'failed_to_stop');
                        oSipStack = null;
                        oSipSessionRegister = null;
                        oSipSessionCall = null;

                        uiOnConnectionEvent(false, false);

                        stopRingbackTone();
                        stopRingTone();

                        uiVideoDisplayShowHide(false);
                        divCallOptions.style.opacity = 0;

                        txtCallStatus.innerHTML = '';
                        txtRegStatus.innerHTML = bFailure ? "<i>Disconnected: <b>" + e.description + "</b></i>" : "<i>Disconnected</i>";
                        break;
                    }

                case 'i_new_call':
                    {
                        if (oSipSessionCall) {
                            // do not accept the incoming call if we're already 'in call'
                            e.newSession.hangup(); // comment this line for multi-line support
                        }
                        else {
                            oSipSessionCall = e.newSession;
                            // start listening for events
                            oSipSessionCall.setConfiguration(oConfigCall);

                            uiBtnCallSetText('Answer');
                            btnHangUp.value = 'Reject';
                            btnCall.disabled = false;
                            btnHangUp.disabled = false;

                            startRingTone();

                            var sRemoteNumber = (oSipSessionCall.getRemoteFriendlyName() || 'unknown');
                            txtCallStatus.innerHTML = "<i>Incoming call from [<b>" + sRemoteNumber + "</b>]</i>";
                            showNotifICall(sRemoteNumber);
                        }
                        break;
                    }

                case 'm_permission_requested':
                    {
                        divGlassPanel.style.visibility = 'visible';
                        break;
                    }
                case 'm_permission_accepted':
                case 'm_permission_refused':
                    {
                        divGlassPanel.style.visibility = 'hidden';
                        if (e.type == 'm_permission_refused') {
                            uiCallTerminated('Media stream permission denied');
                        }
                        break;
                    }

                case 'starting': default: break;
            }
        };

        // Callback function for SIP sessions (INVITE, REGISTER, MESSAGE...)
        function onSipEventSession(e /* SIPml.Session.Event */) {
            tsk_utils_log_info('==session event = ' + e.type);

            switch (e.type) {
                case 'connecting': case 'connected':
                    {
                        var bConnected = (e.type == 'connected');
                        if (e.session == oSipSessionRegister) {
                            uiOnConnectionEvent(bConnected, !bConnected);
                            txtRegStatus.innerHTML = "<i>" + e.description + "</i>";
                        }
                        else if (e.session == oSipSessionCall) {
                            btnHangUp.value = 'HangUp';
                            btnCall.disabled = true;
                            btnHangUp.disabled = false;
                            if (window.btnBFCP) window.btnBFCP.disabled = false;

                            if (bConnected) {
                                stopRingbackTone();
                                stopRingTone();

                                if (oNotifICall) {
                                    oNotifICall.cancel();
                                    oNotifICall = null;
                                }
                            }

                            txtCallStatus.innerHTML = "<i>" + e.description + "</i>";
                            divCallOptions.style.opacity = bConnected ? 1 : 0;

                            if (SIPml.isWebRtc4AllSupported()) { // IE don't provide stream callback
                                uiVideoDisplayEvent(false, true);
                                uiVideoDisplayEvent(true, true);
                            }
                        }
                        break;
                    } // 'connecting' | 'connected'
                case 'terminating': case 'terminated':
                    {
                        if (e.session == oSipSessionRegister) {
                            uiOnConnectionEvent(false, false);

                            oSipSessionCall = null;
                            oSipSessionRegister = null;

                            txtRegStatus.innerHTML = "<i>" + e.description + "</i>";
                        }
                        else if (e.session == oSipSessionCall) {
                            uiCallTerminated(e.description);
                        }
                        break;
                    } // 'terminating' | 'terminated'

                case 'm_stream_video_local_added':
                    {
                        if (e.session == oSipSessionCall) {
                            uiVideoDisplayEvent(true, true);
                        }
                        break;
                    }
                case 'm_stream_video_local_removed':
                    {
                        if (e.session == oSipSessionCall) {
                            uiVideoDisplayEvent(true, false);
                        }
                        break;
                    }
                case 'm_stream_video_remote_added':
                    {
                        if (e.session == oSipSessionCall) {
                            uiVideoDisplayEvent(false, true);
                        }
                        break;
                    }
                case 'm_stream_video_remote_removed':
                    {
                        if (e.session == oSipSessionCall) {
                            uiVideoDisplayEvent(false, false);
                        }
                        break;
                    }

                case 'm_stream_audio_local_added':
                case 'm_stream_audio_local_removed':
                case 'm_stream_audio_remote_added':
                case 'm_stream_audio_remote_removed':
                    {
                        break;
                    }

                case 'i_ect_new_call':
                    {
                        oSipSessionTransferCall = e.session;
                        break;
                    }

                case 'i_ao_request':
                    {
                        if (e.session == oSipSessionCall) {
                            var iSipResponseCode = e.getSipResponseCode();
                            if (iSipResponseCode == 180 || iSipResponseCode == 183) {
                                startRingbackTone();
                                txtCallStatus.innerHTML = '<i>Remote ringing...</i>';
                            }
                        }
                        break;
                    }

                case 'm_early_media':
                    {
                        if (e.session == oSipSessionCall) {
                            stopRingbackTone();
                            stopRingTone();
                            txtCallStatus.innerHTML = '<i>Early media started</i>';
                        }
                        break;
                    }

                case 'm_local_hold_ok':
                    {
                        if (e.session == oSipSessionCall) {
                            if (oSipSessionCall.bTransfering) {
                                oSipSessionCall.bTransfering = false;
                                // this.AVSession.TransferCall(this.transferUri);
                            }
                            txtCallStatus.innerHTML = '<i>Call placed on hold</i>';
                            oSipSessionCall.bHeld = true;
                        }
                        break;
                    }
                case 'm_local_hold_nok':
                    {
                        if (e.session == oSipSessionCall) {
                            oSipSessionCall.bTransfering = false;
                            txtCallStatus.innerHTML = '<i>Failed to place remote party on hold</i>';
                        }
                        break;
                    }
                case 'm_local_resume_ok':
                    {
                        if (e.session == oSipSessionCall) {
                            oSipSessionCall.bTransfering = false;
                            txtCallStatus.innerHTML = '<i>Call taken off hold</i>';
                            oSipSessionCall.bHeld = false;

                            if (SIPml.isWebRtc4AllSupported()) { // IE don't provide stream callback yet
                                uiVideoDisplayEvent(false, true);
                                uiVideoDisplayEvent(true, true);
                            }
                        }
                        break;
                    }
                case 'm_local_resume_nok':
                    {
                        if (e.session == oSipSessionCall) {
                            oSipSessionCall.bTransfering = false;
                            txtCallStatus.innerHTML = '<i>Failed to unhold call</i>';
                        }
                        break;
                    }
                case 'm_remote_hold':
                    {
                        if (e.session == oSipSessionCall) {
                            txtCallStatus.innerHTML = '<i>Placed on hold by remote party</i>';
                        }
                        break;
                    }
                case 'm_remote_resume':
                    {
                        if (e.session == oSipSessionCall) {
                            txtCallStatus.innerHTML = '<i>Taken off hold by remote party</i>';
                        }
                        break;
                    }
                case 'm_bfcp_info':
                    {
                        if (e.session == oSipSessionCall) {
                            txtCallStatus.innerHTML = 'BFCP Info: <i>' + e.description + '</i>';
                        }
                        break;
                    }

                case 'o_ect_trying':
                    {
                        if (e.session == oSipSessionCall) {
                            txtCallStatus.innerHTML = '<i>Call transfer in progress...</i>';
                        }
                        break;
                    }
                case 'o_ect_accepted':
                    {
                        if (e.session == oSipSessionCall) {
                            txtCallStatus.innerHTML = '<i>Call transfer accepted</i>';
                        }
                        break;
                    }
                case 'o_ect_completed':
                case 'i_ect_completed':
                    {
                        if (e.session == oSipSessionCall) {
                            txtCallStatus.innerHTML = '<i>Call transfer completed</i>';
                            if (oSipSessionTransferCall) {
                                oSipSessionCall = oSipSessionTransferCall;
                            }
                            oSipSessionTransferCall = null;
                        }
                        break;
                    }
                case 'o_ect_failed':
                case 'i_ect_failed':
                    {
                        if (e.session == oSipSessionCall) {
                            txtCallStatus.innerHTML = '<i>Call transfer failed</i>';
                        }
                        break;
                    }
                case 'o_ect_notify':
                case 'i_ect_notify':
                    {
                        if (e.session == oSipSessionCall) {
                            txtCallStatus.innerHTML = "<i>Call Transfer: <b>" + e.getSipResponseCode() + " " + e.description + "</b></i>";
                            if (e.getSipResponseCode() >= 300) {
                                if (oSipSessionCall.bHeld) {
                                    oSipSessionCall.resume();
                                }
                            }
                        }
                        break;
                    }
                case 'i_ect_requested':
                    {
                        if (e.session == oSipSessionCall) {
                            var s_message = "Do you accept call transfer to [" + e.getTransferDestinationFriendlyName() + "]?";//FIXME
                            if (confirm(s_message)) {
                                txtCallStatus.innerHTML = "<i>Call transfer in progress...</i>";
                                oSipSessionCall.acceptTransfer();
                                break;
                            }
                            oSipSessionCall.rejectTransfer();
                        }
                        break;
                    }
            }
        }

    </script>
</head>
<body style="cursor:wait">
    <div class="container">
        <div class="row-fluid">
            <div class="span5 well">
                <label align="center" id="txtInfo"> </label>
                <h2> Expert settings</h2>
                <br />
                <table style='width: 100%'>
                    <tr>
                        <td>
                            <label style="height: 100%">Enable RTCWeb Breaker<sup><a href="#aRTCWebBreaker">[1]</a></sup>:</label>
                        </td>
                        <td>
                            <input type='checkbox' id='cbRTCWebBreaker' />
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <label style="height: 100%">WebSocket Server URL<sup><a href="#aWebSocketServerURL">[2]</a></sup>:</label>
                        </td>
                        <td>
                            <input type="text" style="width: 100%; height: 100%" id="txtWebsocketServerUrl" value="" placeholder="e.g. ws://sipml5.org:5062" />
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <label style="height: 100%">SIP outbound Proxy URL<sup><a href="#aSIPOutboundProxyURL">[3]</a></sup>:</label>
                        </td>
                        <td>
                            <input type="text" style="width: 100%; height: 100%" id="txtSIPOutboundProxyUrl" value="" placeholder="e.g. udp://sipml5.org:5060" />
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <label style="height: 100%">ICE Servers<sup><a href="#aIceServers">[4]</a></sup>:</label>
                        </td>
                        <td>
                            <input type="text" style="width: 100%; height: 100%" id="txtIceServers" value="" placeholder="e.g. [{ url: 'stun:stun.l.google.com:19302'}, { url:'turn:user@numb.viagenie.ca', credential:'myPassword'}]" />
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <label style="height: 100%">Max bandwidth (kbps)<sup><a href="#aBandwidth">[5]</a></sup>:</label>
                        </td>
                        <td>
                            <input type="text" style="width: 100%; height: 100%" id="txtBandwidth" value="" placeholder="{ audio:64, video:512 }" />
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <label style="height: 100%">Video size<sup><a href="#aSizeVideo">[6]</a></sup>:</label>
                        </td>
                        <td>
                            <input type="text" style="width: 100%; height: 100%" id="txtSizeVideo" value="" placeholder="{ minWidth: 640, minHeight:480, maxWidth: 640, maxHeight:480 }" />
                        </td>
                    </tr>
                    <tr>
                        <td colspan="2" align="right">
                            <input type="button" class="btn-success" id="btnSave" value="Save" onclick='settingsSave();' />
                            &nbsp;
                            <input type="button" class="btn-danger" id="btnRevert" value="Revert" onclick='settingsRevert();' />
                        </td>
                   </tr>

                </table>
            </div>

            <div class="span5 well">
                <label style="width: 100%;" align="center" id="txtRegStatus">
                </label>
                <h2>
                    Registration
                </h2>
                <br />
                <table style='width: 100%'>
                    <tr>
                        <td>
                            <label style="height: 100%">
                                Display Name:
                            </label>
                        </td>
                        <td>
                            <input type="text" style="width: 100%; height: 100%" id="txtDisplayName" value=""
                                   placeholder="e.g. John Doe" />
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <label style="height: 100%">
                                Private Identity<sup>*</sup>:
                            </label>
                        </td>
                        <td>
                            <input type="text" style="width: 100%; height: 100%" id="txtPrivateIdentity" value=""
                                   placeholder="e.g. +33600000000" />
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <label style="height: 100%">
                                Public Identity<sup>*</sup>:
                            </label>
                        </td>
                        <td>
                            <input type="text" style="width: 100%; height: 100%" id="txtPublicIdentity" value=""
                                   placeholder="e.g. sip:+33600000000@doubango.org" />
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <label style="height: 100%">Password:</label>
                        </td>
                        <td>
                            <input type="password" style="width: 100%; height: 100%" id="txtPassword" value="" />
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <label style="height: 100%">Realm<sup>*</sup>:</label>
                        </td>
                        <td>
                            <input type="text" style="width: 100%; height: 100%" id="txtRealm" value="" placeholder="e.g. doubango.org" />
                        </td>
                    </tr>
                    <tr>
                        <td colspan="2" align="right">
                            <input type="button" class="btn btn-success" id="btnRegister" value="LogIn" disabled onclick='sipRegister();' />
                            &nbsp;
                            <input type="button" class="btn btn-danger" id="btnUnRegister" value="LogOut" disabled onclick='sipUnRegister();' />
                        </td>
                    </tr>
                </table>
            </div>

            <div id="divCallCtrl" class="span10 well" style='display:table-cell; vertical-align:middle'>
                <label style="width: 100%;" align="center" id="txtCallStatus">
                </label>
                <h2>
                    Call control
                </h2>
                <br />
                <table style='width: 100%;'>
                    <tr>
                        <td style="white-space:nowrap;">
                            <input type="text" style="width: 100%; height:100%;" id="txtPhoneNumber" value="" placeholder="Enter phone number to call" />
                        </td>
                    </tr>
                    <tr>
                        <td colspan="1" align="right">
                            <div class="btn-toolbar" style="margin: 0; vertical-align:middle">
                                <div id="divBtnCallGroup" class="btn-group">
                                    <button id="btnCall" disabled class="btn btn-primary" data-toggle="dropdown">Call</button>
                                </div>&nbsp;&nbsp;
                                <div class="btn-group">
                                    <input type="button" id="btnHangUp" style="margin: 0; vertical-align:middle; height: 100%;" class="btn btn-primary" value="HangUp" onclick='sipHangUp();' disabled />
                                </div>
                            </div>
                        </td>
                    </tr>
                    <tr>
                        <td id="tdVideo" class='tab-video'>
                            <div id="divVideo" class='div-video'>
                                <div id="divVideoRemote" style='position:relative; border:1px solid #009; height:100%; width:100%; z-index: auto; opacity: 1'>
                                    <video class="video" width="100%" height="100%" id="video_remote" autoplay="autoplay" style="opacity: 0;
                                        background-color: #000000; -webkit-transition-property: opacity; -webkit-transition-duration: 2s;"></video>
                                </div>

                                <div id="divVideoLocalWrapper" style="margin-left: 0px; border:0px solid #009; z-index: 1000">
                                    <iframe class="previewvideo" style="border:0px solid #009; z-index: 1000"> </iframe>
                                    <div id="divVideoLocal" class="previewvideo" style=' border:0px solid #009; z-index: 1000'>
                                        <video class="video" width="100%" height="100%" id="video_local" autoplay="autoplay" muted="true" style="opacity: 0;
                                            background-color: #000000; -webkit-transition-property: opacity;
                                            -webkit-transition-duration: 2s;"></video>
                                    </div>
                                </div>
                                <div id="divScreencastLocalWrapper" style="margin-left: 90px; border:0px solid #009; z-index: 1000">
                                    <iframe class="previewvideo" style="border:0px solid #009; z-index: 1000"> </iframe>
                                    <div id="divScreencastLocal" class="previewvideo" style=' border:0px solid #009; z-index: 1000'>
                                    </div>
                                </div>
                            </div>
                        </td>
                    </tr>
                    <tr>
                        <td align='center'>
                            <div id='divCallOptions' class='call-options' style='opacity: 0; margin-top: 0px'>
                                <input type="button" class="btn" style="" id="btnMute" value="Mute" onclick='sipToggleMute();' /> &nbsp;
                            </div>
                        </td>
                    </tr>
                </table>
            </div>
        </div>

        <br />
    </div>
    <!-- /container -->
    <!-- Glass Panel -->
    <div id='divGlassPanel' class='glass-panel' style='visibility:hidden'></div>
    <!-- KeyPad Div -->
    <div id='divKeyPad' class='span2 well div-keypad' style="left:0px; top:0px; width:250; height:240; visibility:hidden">
        <table style="width: 100%; height: 100%">
            <tr><td><input type="button" style="width: 33%" class="btn" value="1" onclick="sipSendDTMF('1');" /><input type="button" style="width: 33%" class="btn" value="2" onclick="sipSendDTMF('2');" /><input type="button" style="width: 33%" class="btn" value="3" onclick="sipSendDTMF('3');" /></td></tr>
            <tr><td><input type="button" style="width: 33%" class="btn" value="4" onclick="sipSendDTMF('4');" /><input type="button" style="width: 33%" class="btn" value="5" onclick="sipSendDTMF('5');" /><input type="button" style="width: 33%" class="btn" value="6" onclick="sipSendDTMF('6');" /></td></tr>
            <tr><td><input type="button" style="width: 33%" class="btn" value="7" onclick="sipSendDTMF('7');" /><input type="button" style="width: 33%" class="btn" value="8" onclick="sipSendDTMF('8');" /><input type="button" style="width: 33%" class="btn" value="9" onclick="sipSendDTMF('9');" /></td></tr>
            <tr><td><input type="button" style="width: 33%" class="btn" value="*" onclick="sipSendDTMF('*');" /><input type="button" style="width: 33%" class="btn" value="0" onclick="sipSendDTMF('0');" /><input type="button" style="width: 33%" class="btn" value="#" onclick="sipSendDTMF('#');" /></td></tr>
        </table>
    </div>
    <!-- Call button options -->
    <ul id="ulCallOptions" class="dropdown-menu" style="visibility:hidden">
        <li><a href="#" onclick='sipCall("call-audio");'>Audio</a></li>
        <li><a href="#" onclick='sipCall("call-audiovideo");'>Video</a></li>
        <li id='liScreenShare'><a href="#" onclick='sipShareScreen();'>Screen Share</a></li>
        <li class="divider"></li>
        <li><a href="#" onclick='uiDisableCallOptions();'><b>Disable these options</b></a></li>
    </ul>

    <!-- Le javascript
    ================================================== -->
    <!-- Placed at the end of the document so the pages load faster -->
    <script type="text/javascript" src="./assets/js/jquery.js"></script>
    <script type="text/javascript" src="./assets/js/bootstrap-transition.js"></script>
    <script type="text/javascript" src="./assets/js/bootstrap-alert.js"></script>
    <script type="text/javascript" src="./assets/js/bootstrap-modal.js"></script>
    <script type="text/javascript" src="./assets/js/bootstrap-dropdown.js"></script>
    <script type="text/javascript" src="./assets/js/bootstrap-scrollspy.js"></script>
    <script type="text/javascript" src="./assets/js/bootstrap-tab.js"></script>
    <script type="text/javascript" src="./assets/js/bootstrap-tooltip.js"></script>
    <script type="text/javascript" src="./assets/js/bootstrap-popover.js"></script>
    <script type="text/javascript" src="./assets/js/bootstrap-button.js"></script>
    <script type="text/javascript" src="./assets/js/bootstrap-collapse.js"></script>
    <script type="text/javascript" src="./assets/js/bootstrap-carousel.js"></script>
    <script type="text/javascript" src="./assets/js/bootstrap-typeahead.js"></script>

    <!-- Audios -->
    <audio id="audio_remote" autoplay="autoplay"> </audio>
    <audio id="ringtone" loop src="sounds/ringtone.wav"> </audio>
    <audio id="ringbacktone" loop src="sounds/ringbacktone.wav"> </audio>
    <audio id="dtmfTone" src="sounds/dtmf.wav"> </audio>

</body>
</html>
