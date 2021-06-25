const tf = require('@tensorflow/tfjs-node');

function normalized(data){ // x,y,z
    
    x = (data[0] - 1.262) / 24.18347807
    y = (data[1] - 1.196) / 22.85995322
    z = (data[2] - -248.947) / 23.86685378
    return [x,y,x]

}

function denormalized(data){
    v = (data[0] * 552.6264) + 650.4795
    p = (data[1] * 12153.8) + 10620.5615
    return [v, p]
    
    m1 = (data[0] * 32.59359793) + 10.339
    m2 = (data[1] * 30.54807165) + 9.805
    m3 = (data[2] * 31.36414005) + 10.153
}


async function predict(data){
    let in_dim = 3;
    
    data = normalized(data);
    shape = [1, in_dim];

    tf_data = tf.tensor2d(data, shape);

    try{
        // path load in public access => github
        const path = 'https://raw.githubusercontent.com/KhoirulAziz23/kisi_uas/main/public/ex_model/model.json';
        const model = await tf.loadGraphModel(path);
        
        predict = model.predict(
                tf_data
        );
        result = predict.dataSync();
        return denormalized( result );
        
    }catch(e){
      console.log(e);
    }
}

module.exports = {
    predict: predict 
}
  
