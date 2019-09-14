x <- iris
y <- matrix(0,150,3)
y[1:50,1] <- 1
y[51:100,2] <- 1
y[101:150,3] <- 1
xy <- cbind(x[1:4],y)

#randomize data
xy_acak <- xy[sample(nrow(xy)),]

#input
xin <- xy_acak[1:4]
yt <- xy_acak[5:7]
yt <- as.matrix(yt)
xin <- as.matrix(xin)

#normalisasi data z-score
for (i in 1:4){
  xin[,i] <- (xin[,i]-mean(xin[,i]))/(sd(xin[,i]))
}

#normalisasi data max min [0,1]
for (j in 1:4){
  xin[,j] <- ((xin[,j]-min(xin[,j]))/(max(xin[,j])-min(xin[,j])))
}

N = 4 #jumlah input neuron
M = 8 #jumlah hidden neuron 6,7,8
L = 3 #jumlah output neuron 
alpha = 0.4 #learning rate 0.2 atau 0.4
momentum = 0.7 #momentum 0 atau 0.7

#inisialisasi bobot awal random -0.5 sampai 0,5
v = matrix(runif(N*M),N,M)-0.5 #bobot layer 1
w = matrix(runif(M*L),M,L)-0.5 #bobot layer 2
v0 = matrix(runif(1*M),1)-0.5 #nilai bias layer 1
w0 = matrix(runif(1*L),1)-0.5 #nilai bias layer 2

#fungsi sigmoid biner
sigmoid <- function(x) {
  1/(1 + exp(-x))
}

#turunan fungsi sigmoid biner
sigmoid_d <- function(x) {
  (1/(1 + exp(-x)))*(1-(1/(1 + exp(-x))))
}

#fungsi sigmoid bipolar
sigmoidbip <- function(x) {
  (1-exp(-x))/(1+exp(-x))
}

#turunan fungsi sigmoid bipolar
sigmoidbip_d <- function(x) {
  (2*(exp(-x))/(1+exp(-x))^2)
}

#momentum
dv_old = matrix(0,N,M) #tempat menyimpan delta bobot layer 1
dw_old = matrix(0,M,L) #tempat menyimpan delta bobot layer 1
dv0_old = matrix(0,1,M) #tempat menyimpan delta bobot layer 1
dw0_old = matrix(0,1,L) #tempat menyimpan delta bobot layer 1

#stop condition
epochtot = 2000 #kondisi
errormaks = 0.01 #error maksimal 
epoch = 0 #jumlah epoch awal
error = 10 #nili error cukup besar
errortot = matrix(0,epochtot,1)
yltot = matrix(0,75,3, TRUE)
yluji = matrix(0,75,3, TRUE)

while ((epoch < epochtot) && (error > errormaks)) {
  error = 0
  for (i in 1:75) {
    zin <- xin[i,] %*% v + v0
    zm = sigmoid(zin) #fungsi sigmoid biner
    
    #feedforward layer hidden --> output
    yin <- zm %*% w + w0
    yl <- sigmoid(yin) #fungsi sigmoid biner
    
    #backprop layer output --> hidden
    dl <- sigmoid_d(yin) #turunan fungsi aktivasi sigmoid biner
    deltal <- (yt[i,]-yl)*dl
    deltaw = matrix(0,M,L, TRUE)
    for (m in 1:M){
      for (l in 1:L){
        deltaw[m,l] <- alpha * deltal[1,l] * zm[1,m]
      }
    }
    
    deltaw0 <- alpha * deltal
    
    #backprop layer hidden --> input
    deltain <- deltal %*% t(w)
    dm <- sigmoid_d(zin) #turunan fungsi aktivasi sigmoid biner
    deltam <- deltain*dm
    deltav = matrix(0,N,M, TRUE)
    for (n in 1:N){
      for (m in 1:M){
        deltav[n,m] <- alpha * deltam[1,m] * xin[i,n]
      }
    }
    
    deltav0 <- alpha * deltam
    
    #update bobot dengan momentum
    w <- w + deltaw + momentum * dw_old
    v <- v + deltav + momentum * dv_old
    w0 <- w0 + deltaw0 + momentum * dw0_old
    v0 <- v0 + deltav0 + momentum * dv0_old
    
    #simpan nilai update bobot lama
    dv_old <- deltav
    dw_old <- deltaw
    dv0_old <- deltav0
    dw0_old <- deltaw0
    
    #hitung error
    error <- error + 0.5 * sum ((yt[i,]-yl)^2)
    yltot[i,] <- yl
  }
  #tambah jumlah epoch dengan 1
  epoch <- epoch + 1
  errortot[epoch,1] = error
}

#plot epoch vs error
plot(1:2000, errortot, main = "Epoch vs Error", 
     xlab = "Epoch", ylab = "Sum Squared Error", 
     grid(lty = "dashed" , col =  "black"))

#pengujian untuk setiap data uji
#inisialisasi error
error_uji <- 0
for (i in 76:150){
  #feedforward layer input --> hidden
  zin <- xin[i,] %*% v + v0
  zm <- sigmoid(zin) #fungsi sigmoid biner
  
  #feedforward layer hidden --> output
  yin <- zm %*% w + w0
  yl <- sigmoid(yin) #fungsi sigmoid biner
  
  #rounding hasil klasifikasi dengan treshold nilai maksimum baris yl
  for (k in 1:length(yl)){
    if (yl[k] >= max(yl)) {
      yl[k] = 1
    } else {
      yl[k] = 0
    }
  }
  
  #perhitungan kesalahan klasifikasi
  if (yl[1] != yt[i,1] || yl[2] != yt[i,2] || yl[3] != yt[i,3]) {
    error_uji <- error_uji + 1
  }
  yluji[i-75,] = yl
}

akurasi <- (75-error_uji)/75*100
akurasi
