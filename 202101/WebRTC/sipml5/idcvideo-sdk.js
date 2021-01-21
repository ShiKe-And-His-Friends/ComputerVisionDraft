document.write("<script language='javascript' src='SIPml-api.js'></script>");

// idcMedia.noop = function() {};

function idcMediaInit(callback) {
	
	callback.success = (typeof callback.success == "function") ? callback.success : function() {};
	callback.error = (typeof callback.error == "function") ? callback.error : function() {};

	SIPml.init(()=> {
		if (!SIPml.isWebRtcSupported()) {
		    // is it chrome?
		    if (SIPml.getNavigatorFriendlyName() == 'chrome') {
		    	alert("You're using an old Chrome version or WebRTC is not enabled.\nDo you want to see how to enable WebRTC?");
		        callback.error("You're using an old Chrome version or WebRTC is not enabled.\nDo you want to see how to enable WebRTC?");
		        return;
		    }
		    else {
		    	alert("webrtc-everywhere extension is not installed. Do you want to install it?\nIMPORTANT: You must restart your browser after the installation.");
		    	callback.error("webrtc-everywhere extension is not installed. Do you want to install it?\nIMPORTANT: You must restart your browser after the installation.");
				return;
		    }
		}

		if (!SIPml.isWebSocketSupported()) {
			alert("Your browser don\'t support WebSockets.\nDo you want to download a WebSocket-capable browser?");
		   	callback.error('Your browser don\'t support WebSockets.\nDo you want to download a WebSocket-capable browser?');
		    return;
		}

		if (!SIPml.isWebRtcSupported()) {
			alert("Your browser don\'t support WebRTC.\naudio/video calls will be disabled.\nDo you want to download a WebRTC-capable browser?");
			callback.error('Your browser don\'t support WebRTC.\naudio/video calls will be disabled.\nDo you want to download a WebRTC-capable browser?');
		    return;
		}	
		callback.success();
	});
}


