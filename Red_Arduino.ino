/******************************************************************
 * Emulación red neuronal regresion lineal 16 entradas y 15  neuronas 
 * Autor: 
 *        Juan Camilo Aranda
 *        juan.arandaa@uao.edu.co
 ******************************************************************/
#include <math.h>
/******************************************************************
 * Configuración de la Red
 ******************************************************************/
const int NeuronasOculta = 15;
const int NeuronasEntrada = 16;
const int NeuronasSalida = 1;

//Datos de testeo de la red neuronal, sacados del dataset
const float datosintest[15][16] = {
  
{12.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}, 
{16.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0}, 
{27.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}, 
{24.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}, 
{3.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}, 
{10.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}, 
{1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}, 
{15.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}, 
{3.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}, 
{25.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0}, 
{6.0,2.0,2.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0}, 
{11.0,0.0,2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}, 
{2.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}, 
{16.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}, 
{19.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0} 

  };
const float salidadeseada[1][15] = {
//Salida deseada que queremos de la red neuronal, para graficar la salida de la red y la salida que queremos
{4.7,10.9,19.3,18.3,5.7,8.5,4.1,6.8,5.3,11.8,9.0,11.9,3.8,5.3,11.3}

};
// Estos valores son necesarios para la normalización de los datos cuando entran y salen de la red
// Rango de normalización
const float YMin = 0;
const float YMax = 1;

// Valores para normalizar la entrada de la red, estos valores dependen de los maximos y minimos que tienen cada una de las entradas a la red neuronal
const float X1Min = 1;
const float X1Max = 27;
const float X2Min = 0;
const float X2Max = 4;
const float X3Min = 0;
const float X3Max = 5;
const float X4Min = 0;
const float X4Max = 1;
const float X5Min = 0;
const float X5Max = 3;
const float X6Min = 0;
const float X6Max = 2;
const float X7Min = 0;
const float X7Max = 1;
const float X8Min = 0;
const float X8Max = 1;
const float X9Min = 0;
const float X9Max = 1;
const float X10Min = 0;
const float X10Max = 4;
const float X11Min = 0;
const float X11Max = 1;
const float X12Min = 0;
const float X12Max = 7;
const float X13Min = 0;
const float X13Max = 1;
const float X14Min = 0;
const float X14Max = 8;
const float X15Min = 0;
const float X15Max = 1;
const float X16Min = 0;
const float X16Max = 4;




// Valores para desnormalizar la salida de la red
const float DMin =3.4;
const float DMax =23.4;



// Se extraen los Pesos capa oculta de la red neuronal creada en google colab y los pegamos en la siguiente matrix
const float PesosOculta[NeuronasOculta][NeuronasEntrada+1]= {
{-0.24419254,-0.08349195,0.058465272,-0.40628555,-0.16931805,-0.39448363,0.4000541,0.008324176,0.20987761,-0.2776457,-0.021373361,-0.42835215,-0.040771753,-0.17337781,0.27484608,-0.25519374,0.0}, 
{-0.22527611,-0.38676128,0.040579706,-0.11892545,-0.43656284,0.43528152,-0.15271476,0.39913374,-0.38424298,-0.31408006,0.37064457,0.4274434,0.27503216,0.05863418,0.049633205,0.14757474,-0.005114303}, 
{0.63671994,0.22402285,0.24127981,-0.24950284,0.07908652,-0.7454926,-0.108766526,-0.28288183,-0.34733406,0.2209781,-0.45009875,0.011456503,0.4940414,-0.07894716,-0.031094115,-0.1534053,0.083569124}, 
{0.12217518,-0.17085922,-0.05111055,-0.1815039,-0.3610844,0.18781678,0.41610318,-0.21098058,0.21488221,-0.18257485,0.3927471,0.07896639,-0.04698541,0.030449796,0.047614172,0.04876039,-0.12564366}, 
{0.6205396,0.24369268,-0.42794156,-2.078966,-0.1426571,0.85728633,0.09820908,0.7468882,0.45728114,-0.37023973,-0.11250323,0.01850441,-0.5483118,-0.69203347,2.1044579,0.2511009,-0.06909331}, 
{-0.11773652,-0.005000651,-0.37290844,-0.42205086,-0.31005508,0.3543216,0.19784999,0.28222948,-0.27201006,0.27674752,-0.075895816,-0.014453441,-0.18334165,-0.03194201,0.3848585,-0.2456073,0.0}, 
{0.47963798,-0.4408987,0.69721705,-2.7546806,0.71694785,-4.76001,0.19951612,0.62731177,0.32499942,-0.65371037,0.20956227,-0.48481828,-2.5815442,-1.833936,-0.7747203,0.18968426,-0.03662786}, 
{0.25349045,0.56252444,0.119195506,0.18562222,-0.6211057,-0.24447647,-0.15506399,-0.20662162,-0.85364145,0.10852096,-0.14977713,0.102174915,0.41328618,-4.291088,-1.3430899,0.51399374,-0.0097731715}, 
{0.5747624,0.32711574,0.11287773,-0.55248135,0.50471467,-0.7443199,0.3853914,-0.27975002,-0.7762567,0.033603083,-0.46954274,-0.14435053,0.48613885,-0.23474765,-0.4097659,0.44017273,0.07633185}, 
{-0.011840936,0.047375675,-0.19154583,-0.003885448,-0.14308468,-0.027639236,-0.06053853,-0.34829637,0.07509102,0.37395597,0.18302822,-0.082965285,0.36847332,-0.34271097,-0.37421772,-0.08750591,-0.03672227}, 
{0.24495225,-0.29642525,-0.1748761,-0.3941303,0.39103347,-0.48459065,-0.20592462,-0.807857,-0.32250655,0.05022004,-0.28476974,0.12733328,0.04251387,-0.15740663,0.16037717,0.6496365,0.050698258}, 
{0.213839,-0.25845116,0.45492333,-0.109723,0.11879284,-0.34592155,0.3594852,-0.73195004,-0.014868718,0.5274863,0.041280534,0.06696288,-0.16143304,0.02131786,-0.3747297,0.2894596,0.07499538}, 
{-0.04142731,0.43090066,-0.47057942,0.6239073,0.27268282,0.89957917,-0.24903344,0.6563885,0.529567,-0.84758544,-0.13159241,-2.9913251,1.7804083,0.040872775,-0.02510127,-0.51161724,0.2364389}, 
{0.5589313,-0.18885598,0.08917646,-0.39092827,0.22912025,-0.5340639,-0.303773,-0.73284924,-0.59152925,-0.14705674,-0.23586492,0.2810349,0.23306549,-0.16816089,0.192982,0.4822027,0.09011274}, 
{-0.007679014,-0.29727763,0.24199516,0.124608085,0.19640286,0.015890058,0.35829872,0.060286146,-0.05257617,0.3378209,0.08347173,-0.39760348,0.15268421,-0.23499292,-0.27162874,-0.086661465,-0.04816628}

};


// Se extraen los Pesos de la capa de salida de la red neuronal creada en google colab y los pegamosen la siguiente matrix
const float PesosSalida[NeuronasSalida][NeuronasOculta+1] = {
{0.34495312,-0.103217684,0.6035414,-0.43458486,-0.72666216,-0.3450349,-1.3730055,0.7127101,0.46269715,-0.45912987,0.22973816,0.5918975,-0.6839539,0.85648763,-0.4206157,0.1058427}

};  

//Definimos el variables, numero de neuronas en la capa oculta, capa de salida y de entrada
int i, j, p, q, r;
float Neta;
float CapaOculta[NeuronasOculta];
float CapaSalida[NeuronasSalida];
float CapaEntrada[NeuronasEntrada];
int contador = 0;
//Iniciamos la comunicacion serial para graficar
void setup(){
  //start serial connection
  Serial.begin(9600);
  randomSeed(analogRead(0));
}

//inicio del bucle
void loop(){
 if (contador > 10){
  contador = 0;
  }
 float Entrada;
 float Salida;

// Normalización de las entradas que se usará en la red neuronal
 
CapaEntrada[0]=YMin+ ((datosintest[contador][0]-X1Min)*((YMax-YMin)/(X1Max-X1Min)));
CapaEntrada[1]=YMin+ ((datosintest[contador][1]-X2Min)*((YMax-YMin)/(X2Max-X2Min)));
CapaEntrada[2]=YMin+ ((datosintest[contador][2]-X3Min)*((YMax-YMin)/(X3Max-X3Min)));
CapaEntrada[3]=YMin+ ((datosintest[contador][3]-X4Min)*((YMax-YMin)/(X4Max-X4Min)));
CapaEntrada[4]=YMin+ ((datosintest[contador][4]-X5Min)*((YMax-YMin)/(X5Max-X5Min)));
CapaEntrada[5]=YMin+ ((datosintest[contador][5]-X6Min)*((YMax-YMin)/(X6Max-X6Min)));
CapaEntrada[6]=YMin+ ((datosintest[contador][6]-X7Min)*((YMax-YMin)/(X7Max-X7Min)));
CapaEntrada[7]=YMin+ ((datosintest[contador][7]-X8Min)*((YMax-YMin)/(X8Max-X8Min)));
CapaEntrada[8]=YMin+ ((datosintest[contador][8]-X9Min)*((YMax-YMin)/(X9Max-X9Min)));
CapaEntrada[9]=YMin+ ((datosintest[contador][9]-X10Min)*((YMax-YMin)/(X10Max-X10Min)));
CapaEntrada[10]=YMin+ ((datosintest[contador][10]-X11Min)*((YMax-YMin)/(X11Max-X11Min)));
CapaEntrada[11]=YMin+ ((datosintest[contador][11]-X12Min)*((YMax-YMin)/(X12Max-X12Min)));
CapaEntrada[12]=YMin+ ((datosintest[contador][12]-X13Min)*((YMax-YMin)/(X13Max-X13Min)));
CapaEntrada[13]=YMin+ ((datosintest[contador][13]-X14Min)*((YMax-YMin)/(X14Max-X14Min)));
CapaEntrada[14]=YMin+ ((datosintest[contador][14]-X15Min)*((YMax-YMin)/(X15Max-X15Min)));
CapaEntrada[15]=YMin+ ((datosintest[contador][15]-X16Min)*((YMax-YMin)/(X16Max-X16Min)));

/******************************************************************
* Calculo de la salida de la capa oculta
******************************************************************/
    for( i = 0 ; i < NeuronasOculta ; i++ ) {    
      Neta = PesosOculta[i][NeuronasEntrada] ;
      for( j = 0 ; j < NeuronasEntrada ; j++ ) {
        Neta += PesosOculta[i][j]*CapaEntrada[j];
      }
//      CapaOculta[i] = (2.0/(1.0 + exp(-2*Neta)))-1.0; 
      if(Neta < 0){
        CapaOculta[i] = 0;
        
        }
      else{
        CapaOculta[i] = Neta;
        }
      
    }
/******************************************************************
* Calculo de la salida de la red
******************************************************************/
    for( i = 0 ; i < NeuronasSalida ; i++ ) {    
      Neta = PesosSalida[i][NeuronasOculta] ;
      for( j = 0 ; j < NeuronasOculta ; j++ ) {
        Neta +=  PesosSalida[i][j]*CapaOculta[j];
      }

    
    if(Neta < 0){
      CapaSalida[i] = 0;      
      }
    else{
      CapaSalida[i] = Neta; 
      }
     
    }
// La salida da la red esta nomalizada, para que quede en el rango original, es necesario desnormalizar
// Desnormalización de la salida de la red neuronal
Salida=DMin+ ((CapaSalida[0]-YMin)*((DMax-DMin)/(YMax-YMin)));

//Enviamos los datos por comunicacion serial
Serial.print("red:");     
Serial.println(Salida);  
Serial.print(",");  
Serial.print("deseada:");     
Serial.println(salidadeseada[0][contador]);

contador++;
delay(1000);  
}
