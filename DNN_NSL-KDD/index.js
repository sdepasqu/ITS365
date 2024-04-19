async function runExample() {
    var x = [];

    for (let i = 1; i <= 41; i++) {
        x.push(parseFloat(document.getElementById('box' + i).value));
    }

    let tensorX = new onnx.Tensor(x, 'float32', [1, 41]);

    let session = new onnx.InferenceSession();

    await session.loadModel("https://sdepasqu.github.io/ITS365/DNN_NSL-KDD/DNN_NSL-KDD.onnx");
    let outputMap = await session.run([tensorX]);
    let outputData = outputMap.get('output1');

    let predictions = document.getElementById('predictions');

    if (outputData) {
        let data = outputData.data;
        if (data && data.length > 0) {
            predictions.innerHTML = ` <hr> Got an output tensor with values: <br/>
                <table>
                <tr>
                <td>  Is it an attack?  </td>
                <td id="td0">  ${data[0].toFixed(2)}  </td>
                </tr>
                </table>`;
        } else {
            predictions.innerHTML = "Output data is empty.";
        }
    } else {
        predictions.innerHTML = "Output data is undefined.";
    }
}
