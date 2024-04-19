async function runExample() 
{
    var x = [];

    for (let i = 1; i <= 41; i++) 
    {
        x.push(parseFloat(document.getElementById('box' + i).value));
    }

    let tensorX = new onnx.Tensor(x, 'float32', [1, 41]);

    let session = new onnx.InferenceSession();

    await session.loadModel("/home/salvatoredepasquale/Documents/School/ITS365/365project/DNN_NSL-KDD.onnx");
    let outputMap = await session.run([tensorX]);
    let outputData = outputMap.get('output1');

    let predictions = document.getElementById('predictions');

    predictions.innerHTML = ` <hr> Got an output tensor with values: <br/>
    <table>
    <tr>
    <td>  Rating of Wine Quality  </td>
    <td id="td0">  ${outputData.data[0].toFixed(2)}  </td>
    </tr>
    </table>`;
}