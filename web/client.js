var pc = null;

function getElement(id) {
    return document.getElementById(id);
}

function ensureMediaElement(id, tagName) {
    var element = getElement(id);
    if (!element) {
        element = document.createElement(tagName);
        element.id = id;
        element.autoplay = true;
        if (tagName === 'video') {
            element.playsInline = true;
        }
        element.style.display = 'none';
        document.body.appendChild(element);
    }
    return element;
}

function setSessionId(sessionid) {
    var input = getElement('sessionid');
    if (input) {
        input.value = sessionid;
    }
}

function toggleButtons(isConnected) {
    var startButton = getElement('start');
    var stopButton = getElement('stop');

    if (startButton) {
        startButton.style.display = isConnected ? 'none' : 'inline-block';
    }
    if (stopButton) {
        stopButton.style.display = isConnected ? 'inline-block' : 'none';
    }
}

function notifyWebRTCError(error) {
    if (typeof window.onWebRTCError === 'function') {
        window.onWebRTCError(error);
        return;
    }
    alert(error.message || String(error));
}

function bindPeerEvents() {
    var currentPc = pc;

    currentPc.addEventListener('track', function(evt) {
        if (evt.track.kind === 'video') {
            ensureMediaElement('video', 'video').srcObject = evt.streams[0];
        } else {
            var audio = ensureMediaElement('audio', 'audio');
            audio.srcObject = evt.streams[0];
            var playPromise = audio.play();
            if (playPromise && typeof playPromise.catch === 'function') {
                playPromise.catch(function() {});
            }
        }
    });

    currentPc.addEventListener('connectionstatechange', function() {
        if (currentPc.connectionState === 'connected') {
            if (typeof window.onWebRTCConnected === 'function') {
                window.onWebRTCConnected();
            }
        } else if (['failed', 'closed', 'disconnected'].includes(currentPc.connectionState)) {
            if (typeof window.onWebRTCDisconnected === 'function') {
                window.onWebRTCDisconnected(currentPc.connectionState);
            }
        }
    });
}

function waitForIceGathering(peerConnection, timeoutMs) {
    return new Promise(function(resolve) {
        if (!peerConnection || peerConnection.iceGatheringState === 'complete') {
            resolve();
            return;
        }

        var finished = false;
        var timeoutId = null;
        var finish = function() {
            if (finished) {
                return;
            }
            finished = true;
            if (timeoutId) {
                window.clearTimeout(timeoutId);
            }
            peerConnection.removeEventListener('icegatheringstatechange', handleStateChange);
            resolve();
        };
        var handleStateChange = function() {
            if (peerConnection.iceGatheringState === 'complete') {
                finish();
            }
        };

        peerConnection.addEventListener('icegatheringstatechange', handleStateChange);
        if (timeoutMs > 0) {
            timeoutId = window.setTimeout(finish, timeoutMs);
        }
    });
}

function negotiate(extraPayload, options) {
    var currentPc = pc;
    var iceGatheringTimeoutMs = options && typeof options.iceGatheringTimeoutMs === 'number'
        ? options.iceGatheringTimeoutMs
        : 0;

    currentPc.addTransceiver('video', { direction: 'recvonly' });
    currentPc.addTransceiver('audio', { direction: 'recvonly' });

    return currentPc.createOffer()
        .then(function(offer) {
            return currentPc.setLocalDescription(offer);
        })
        .then(function() {
            return waitForIceGathering(currentPc, iceGatheringTimeoutMs);
        })
        .then(function() {
            var offer = currentPc.localDescription;
            var payload = Object.assign({
                sdp: offer.sdp,
                type: offer.type
            }, extraPayload || {});
            var headers = {
                'Content-Type': 'application/json'
            };
            var token = typeof window.getAuthToken === 'function' ? window.getAuthToken() : '';
            if (token) {
                headers['Authorization'] = 'Bearer ' + token;
            }
            return fetch('/offer', {
                body: JSON.stringify(payload),
                headers: headers,
                method: 'POST'
            });
        })
        .then(function(response) {
            if (!response.ok) {
                throw new Error('WebRTC 协商失败: ' + response.status);
            }
            return response.json();
        })
        .then(function(answer) {
            setSessionId(answer.sessionid || 0);
            return currentPc.setRemoteDescription({
                sdp: answer.sdp,
                type: answer.type
            }).then(function() {
                return answer;
            });
        });
}

function start(extraPayload) {
    if (pc && pc.connectionState && pc.connectionState !== 'closed') {
        stop(true);
    }

    var config = {
        sdpSemantics: 'unified-plan'
    };
    var stunCheckbox = getElement('use-stun');
    var iceGatheringTimeoutMs = 200;
    if (stunCheckbox && stunCheckbox.checked) {
        config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];
        iceGatheringTimeoutMs = 1200;
    }

    pc = new RTCPeerConnection(config);
    bindPeerEvents();
    toggleButtons(true);

    return negotiate(extraPayload, {
        iceGatheringTimeoutMs: iceGatheringTimeoutMs
    }).catch(function(error) {
        stop(true);
        notifyWebRTCError(error);
        throw error;
    });
}

function stop(silent) {
    var currentPc = pc;
    pc = null;
    toggleButtons(false);
    setSessionId(0);

    var video = getElement('video');
    var audio = getElement('audio');
    if (video) {
        video.srcObject = null;
    }
    if (audio) {
        audio.srcObject = null;
    }

    if (currentPc) {
        try {
            currentPc.close();
        } catch (error) {
            console.error(error);
        }
    }

    if (!silent && typeof window.onWebRTCDisconnected === 'function') {
        window.onWebRTCDisconnected('manual');
    }
}

window.start = start;
window.stop = stop;
window.getPeerConnection = function() {
    return pc;
};

window.onunload = function() {
    stop(true);
};

window.onbeforeunload = function() {
    stop(true);
};