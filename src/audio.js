<input type = "file" accept = "audio/*" id = "audioFile">
<audio id="audiobook" controls></audio>
<label for="speedSelect">Playback speed:</label>
<select id="speedSelect">
    <option value="0.8">0.8x</option> //slowest value
    <option value="0.9">0.9x</option> //second slowest value
    <option value="1.0">1.0x</option> //normal speed
</select>
<p id="attentionStatus">
    Choose an audio file to play:
</p>
<p id="speedStatus">
    Current playback speed: 1.0x
</p>
</input>
