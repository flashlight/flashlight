"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

Here are some of the possible references:
https://blog.addpipe.com/recording-audio-in-the-browser-using-pure-html5-and-minimal-javascript/
https://stackoverflow.com/a/18650249
https://hacks.mozilla.org/2014/06/easy-audio-capture-with-the-mediarecorder-api/
https://air.ghost.io/recording-to-an-audio-file-using-html5-and-js/
https://stackoverflow.com/a/49019356
"""
from base64 import b64decode

import ffmpeg
import sox
from google.colab.output import eval_js
from IPython.display import HTML, display


AUDIO_HTML = """
<script>
var recordButton = document.createElement("BUTTON");
recordButton.appendChild(
  document.createTextNode("Press to start recording")
);
restyleButtonBeforeRecording();

var my_div = document.createElement("DIV");
my_div.appendChild(recordButton);

document.body.appendChild(my_div);

var base64data = 0;
var reader;
var recorder, gumStream;

function restyleButtonBeforeRecording() {
  recordButton.style.width = '270px';
  recordButton.style.height = '90';
  recordButton.style.padding = '25px';
  recordButton.style.backgroundColor = '#4CAF50';
  recordButton.style.fontSize = '18px';
}

function restyleButtonForRecording() {
  recordButton.style.backgroundColor = '#008CBA';
  recordButton.innerText = "Recording... press to stop";
}

function restyleButtonForSaving() {
  recordButton.style.backgroundColor = '#b34d4d';
  recordButton.innerText = "Saving... please wait!"
}

var handleSuccess = function(stream) {
  gumStream = stream;
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = function(e) {
    var url = URL.createObjectURL(e.data);
    var preview = document.createElement('audio');
    preview.controls = true;
    preview.src = url;
    document.body.appendChild(preview);

    reader = new FileReader();
    reader.readAsDataURL(e.data);
    reader.onloadend = function() {
      base64data = reader.result;
      //console.log("Inside FileReader:" + base64data);
    }
  };
  recorder.start();
  };


function toggleRecording() {
  if (recorder && recorder.state == "recording") {
      recorder.stop();
      gumStream.getAudioTracks()[0].stop();
      restyleButtonForSaving();
  }
}

// https://stackoverflow.com/a/951057
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

var data = new Promise(resolve=>{
  recordButton.onclick = () => {
    restyleButtonForRecording();
    recordButton.onclick = () => {
      toggleRecording();
      sleep(2000).then(() => {
        // wait 2000ms for the data to be available...
        // ideally this should use something like await...
        // console.log("Inside data:" + base64data)
        resolve(base64data.toString());
      });
    };
    navigator.mediaDevices.getUserMedia({audio: true}).then(handleSuccess);
  };
});

</script>
"""


def convert(inputfile, outfile):
    sox_tfm = sox.Transformer()
    sox_tfm.set_output_format(
        file_type="wav", channels=1, encoding="signed-integer", rate=16000, bits=16
    )
    sox_tfm.build(inputfile, outfile)


def record_audio(filename):
    display(HTML(AUDIO_HTML))
    data = eval_js("data")
    binary = b64decode(data.split(",")[1])

    process = (
        ffmpeg.input("pipe:0")
        .output(filename + ".48kHz.wav", format="wav")
        .run_async(
            pipe_stdin=True,
            pipe_stdout=True,
            pipe_stderr=True,
            quiet=True,
            overwrite_output=True,
        )
    )
    output, err = process.communicate(input=binary)
    if process.returncode != 0:
        print("Error during recording")
    convert(filename + ".48kHz.wav", filename + ".wav")
