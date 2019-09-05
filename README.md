test_image_0.jpg 7
test_image_1.jpg 2
test_image_2.jpg 1
test_image_3.jpg 0
test_image_4.jpg 4
test_image_5.jpg 1
test_image_6.jpg 4
test_image_7.jpg 9
test_image_8.jpg 5
test_image_9.jpg 9

# curl test examples
curl -F "image=@images/test_image_8.jpg" http://127.0.0.1:5000/mnist
