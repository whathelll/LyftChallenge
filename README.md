# Standard UNet
1200/1200 [==============================] - 181s 151ms/step - loss: 0.0332 - Lyft_FScore: 0.9668 - Lyft_car_Fscore: 0.9388 - Lyft_road_Fscore: 0.9948  
Training cycle took: 32.608065617084506  
1000/1000 [==============================] - 40s 40ms/step  
Validation: loss: 0.0968 - Lyft_FScore: 0.9032 - Lyft_car_Fscore: 0.8222 - Lyft_road_Fscore: 0.9843  

1200/1200 [==============================] - 180s 150ms/step - loss: 0.0359 - Lyft_FScore: 0.9641 - Lyft_car_Fscore: 0.9332 - Lyft_road_Fscore: 0.9950  
Training cycle took: 32.555992682774864  
1000/1000 [==============================] - 39s 39ms/step  
Validation: loss: 0.0960 - Lyft_FScore: 0.9040 - Lyft_car_Fscore: 0.8253 - Lyft_road_Fscore: 0.9827

1200/1200 [==============================] - 180s 150ms/step - loss: 0.0378 - Lyft_FScore: 0.9622 - Lyft_car_Fscore: 0.9295 - Lyft_road_Fscore: 0.9949  
Training cycle took: 32.53776381413142  
1000/1000 [==============================] - 40s 40ms/step  
Validation: loss: 0.0950 - Lyft_FScore: 0.9050 - Lyft_car_Fscore: 0.8251 - Lyft_road_Fscore: 0.9849

# Unet with spatial convolutions
1200/1200 [==============================] - 195s 162ms/step - loss: 0.0361 - Lyft_FScore: 0.9639 - Lyft_car_Fscore: 0.9337 - Lyft_road_Fscore: 0.9941  
Training cycle took: 34.99065540631612  
1000/1000 [==============================] - 44s 44ms/step  
loss: 0.0949 - Lyft_FScore: 0.9051 - Lyft_car_Fscore: 0.8269 - Lyft_road_Fscore: 0.9833




# Speed comparison
### model_v4
CPU:
eager: 0.3854
graph: 0.3202
graph optimized: 0.3137

GPU:
eager: 0.0435
graph: 0.0172
graph optimized: 0.0154

### UNet
CPU: 
eager: 1.7789
graph: 1.7392
graph optimized: 1.7416

GPU:
eager: 0.0494
graph: 0.0444
graph optimized: 0.041


### More proper ENet
GPU: 
eager: 0.0623
graph: 0.0213
graph optimized: 0.019
