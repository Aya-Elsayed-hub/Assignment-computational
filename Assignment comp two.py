w6 = 0.42
w7 = 0.53
w8 = 0.58
 
eta = 0.5  

 
 
dE_dw6 = 0.02267  # Example value, update as needed
dE_dw7 = 0.0374   # Example value, update as needed
dE_dw8 = 0.03726  # Example value, update as needed

 
 
w6_new = w6 - eta * dE_dw6
w7_new = w7 - eta * dE_dw7
w8_new = w8 - eta * dE_dw8
 
print("Updated weights:")
 
print("w6:", w6_new)
print("w7:", w7_new)
print("w8:", w8_new)



