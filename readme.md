This is a PIDN designed based on PINNs.
By fitting derivative values instead of function values, our method can stably solve the Blasius equation over longer intervals,
whereas the PINNs method often fails to fit correctly over such long intervals.
 Our method, PIDN, has successfully fitted the Blasius equation over intervals ranging from 1000 to 3000, without any involvement of numerical solutions,
  requiring only initial and boundary conditions.
PINNs
![PINNs for Blasius on small domain ](images/comparison_plot_PINNs.png)


XPINNs
![XPINNs for Blasius on small domain ](images/comparison_plot_xpinns.png)


PIDNs
![PIDNs for Blasius on large domain ](images/pidn_1000_function.png)
![PIDNs for Blasius on large domain ](images/pidn_1000_derivative.png)
