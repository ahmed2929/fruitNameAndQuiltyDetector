const fs=require('fs');
const path=require('path')
const  tf =require("@tensorflow/tfjs-node");
require('dotenv').config()
const classes={
    0: "acerolas",
    1: "apples",
    2: "apricots",
    3: "avocados",
    4: "bananas",
    5: "blackberries",
    6: "blueberries",
    7: "cantaloupes",
    8: "cherries",
    9: "coconuts",
    10: "figs",
    11: "grapefruits",
    12: "grapes",
    13: "guava",
    14: "kiwifruit",
    15: "lemons",
    16: "limes",
    17: "mangos",
    18: "olives",
    19: "oranges",
    20: "passionfruit",
    21: "peaches",
    22: "pears",
    23: "pineapples",
    24: "plums",
    25: "pomegranates",
    26: "raspberries",
    27: "strawberries",
    28: "tomatoes",
    29: "watermelons"
  };
  processImage = async (image) => {
    return tf.tidy(() => image.expandDims(0).toFloat().div(127).sub(1));
  }

  module.exports =async (imagePath)=>{
    const modelURL = process.env.hostName+'model/fruits/model.json';;
    const mobilenet = await tf.loadLayersModel(modelURL);
    const buffer=fs.readFileSync(imagePath)
    const image = tf.tidy( () => tf.node.decodeImage(buffer))
    const imageData = await processImage(image);

    //mobilenet.fit(tf.stack(image))

  const prediction =mobilenet.predict(imageData)
  const index = await prediction
  .as1D()
  .argMax()
  .data();
const confidence = await prediction.as1D().data();


return {
  prdiction:classes[index],
  confidence:confidence[index]}
  }





// var storage=multer.diskStorage({
//     destination:(req,file,cb)=>{
//         cb(null,'uploads/images')
//     },
//     filename:(req,file,cb)=>{
//     cb(null,file.fieldname+'-'+Date.now()+path.extname(file.originalname))
    
    
    
//     }
    
    
    
//     })
    
//     var checkImage=function(file,cb){
    
    
//     var ext=path.extname(file.originalname);
    
//     if(ext==='.png'||ext==='.jpg'||ext==='.jpeg'){
//         cb(null,true)
//     }else{
//         cb('not an image',false)
//     }
    
    
//     }
    
    
//     var upload=multer({
//         storage:storage,
//         fileFilter:function(req,file,cb){
//             checkImage(file,cb)
//         }
//     })

//     app.use(express.static('model'))

// app.post("/classify/image",upload.any('img'),async(req,res,next)=>{
//     console.debug(req.files[0].path)
//  //  await sharp(req.files[0].path).resize({ height:150, width:150}).toFile(req.files[0].path)

//    const image= fs.readFileSync(req.files[0].path)
   
    
//     res.json({
//         result
//     })

// })
// // app.get("*",(req,res)=>{
// //     res.send("welcome to fresh fruit project")
// // })
// app.listen(8080)