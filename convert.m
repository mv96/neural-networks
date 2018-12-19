function check=convert(pred)
  m=size(pred,1);
  [rows_1 coloumns_1]=find(pred==1);
  [rows_2 coloumns_2]=find(pred==2);
  pred(rows_1)=0;
  pred(rows_2)=1;
  check=pred;
  end