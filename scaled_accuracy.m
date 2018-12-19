function scaled_accuracy=rescale_detection(accuracy)
  %this function converts the accuracy into a scale down feature where the min is the best value 
  %which has accuracy close to 100
  scaled_accuracy=abs(accuracy-100);
  end