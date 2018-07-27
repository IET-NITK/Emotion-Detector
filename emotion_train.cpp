#include "template.h"
// #include <opencv2/opencv.hpp>
// using namespace cv;
#define vvvld vector < vector < vector <ld> > >
#define vvld vector <  vector <ld> >
#define vld vector < ld >
/*
model : 6 layers
        1st layer : i/p : 48*48*1
        2nd layer : 5*5*1 conv(p=1,s=1) with 10 filters : 46*46*10
        3rd layer : 5*5*10 conv(p=1,s=1) with 10 filters : 44*44*10
        4th layer : max pool (/2): 22*22*10
        5th layer : 5*5*10 conv(p=1,s=2) with 7 filters : 10*10*7 = FC 700
        6th layer : o/p
*/
#define relu(x) ((ld)(x>0)*x)
#define sigmoid(x) (1/(1+exp(-x)))
#define activation(x) sigmoid(x)

#define number_of_images 4178
// int train[48][48][4178];
vvvld train(4178,vvld (48,vld (48)));
int label[4178];
short train_size=0.75*4178.0;

vvvld network[7];
ld w1[5][5][1][10],w2[5][5][10][10],w4[5][5][10][7],wFC[700][7];
void random_initialize(){
    
    srand(time(NULL));
    short i,j,k,u;
    f(i,5)
        f(j,5)
            f(k,1)
                f(u,10)
                    w1[i][j][k][u]=((ld)(rand()-rand()))/1000000.0;
    f(i,5)
        f(j,5)
            f(k,10)
                f(u,10)
                    w2[i][j][k][u]=((ld)(rand()-rand()))/1000000.0;
    f(i,5)
        f(j,5)
            f(k,10)
                f(u,7)
                    w4[i][j][k][u]=((ld)(rand()-rand()))/1000000.0;
    f(i,700)
        f(j,7)
            wFC[i][j]=((ld)(rand()-rand()))/1000000.0;
}

void forward_prop(int image_no){
    network[0][0] = train[image_no];
    
    vvvld temp;
    short i,j,k;

    //network[1]
    temp.resize(1,vvld(50,vld(50,0)));
    f(i,50)
        f(j,50){
            temp[0][i][j]=0;
            if((i-1)>=0&&(j-1)>=0&&(i-1)<48&&(j-1)<48)
                temp[0][i][j]=network[0][0][i-1][j-1];
        }

    f(i,46){
        f(j,46){
            f(k,10){
                short u;
                network[1][k][i][j]=0;
                f(u,25){
                    network[1][k][i][j]+=w1[u/5][u%5][0][k]*temp[0][i+(u/5)][j+(u%5)];
                }
                network[1][k][i][j] = activation(network[1][k][i][j]);
            }
        }
    }

    // network 2
    temp.resize(10,vvld(48,vld(48,0)));
    f(k,10)
    f(i,48)
        f(j,48){
            temp[k][i][j]=0;
            if(i-1>=0&&j-1>=0&&i-1<48&&j-1<48)
                temp[k][i][j]=network[1][k][i-1][j-1];
        }

    f(i,44){
        f(j,44){
            f(k,10){
                short u,u2;
                network[2][k][i][j]=0;
                f(u2,10)
                    f(u,25){
                        network[2][k][i][j]+=w2[u/5][u%5][u2][k]*temp[u2][i+(u/5)][j+(u%5)];
                    }
                network[2][k][i][j] = activation(network[2][k][i][j]);
            }
        }
    }

    // network 3
    f(i,22){
        f(j,22){
            f(k,10){
                network[3][k][i][j]=max(max(network[2][k][2*i][2*j],network[2][k][2*i+1][2*j]),max(network[2][k][2*i][2*j+1],network[2][k][2*i+1][2*j+1]));
            }
        }
    }

    // network 4
    temp.resize(10,vvld(24,vld(24,0)));
    f(k,10)
    f(i,24)
        f(j,24){
            temp[k][i][j]=0;
            if(i-1>=0&&j-1>=0&&i-1<48&&j-1<48)
                temp[k][i][j]=network[3][k][i-1][j-1];
        }

    f(i,10){
        f(j,10){
            f(k,7){
                short u,u2;
                network[4][k][i][j]=0;
                f(u2,10)
                    f(u,25){
                        network[4][k][i][j]+=w4[u/5][u%5][u2][k]*temp[u2][(2*i)+(u/5)][(2*j)+(u%5)];
                    }
                network[4][k][i][j] = activation(network[4][k][i][j]);
            }
        }
    }

    // network 5 or o/p
    f(i,7){
        network[5][0][0][i]=0;
        f(j,700)
            network[5][0][0][i]+=network[4][j/100][(j/10)%10][j%10]*wFC[j][i];
        network[5][0][0][i] = activation(network[5][0][0][i]);
    }
}
void back_prop(){

    // network
    network[0].resize(1,vvld(48,vld(48,0)));
    network[1].resize(10,vvld(46,vld(46,0)));
    network[2].resize(10,vvld(44,vld(44,0)));
    network[3].resize(10,vvld(22,vld(22,0)));
    network[4].resize(7,vvld(10,vld(10,0))); // FC 700
    network[5].resize(1,vvld(1,vld(7,0)));


    short i,j,number_of_iterations=20;
    // random initialisation
    random_initialize();

    // do the back prop
    while(number_of_iterations--){
        short image_no,pred_label[4178];
        double accuracy = 0;
        vvvld error[7];
        error[0].resize(1,vvld(48,vld(48,0))); // useless
        error[1].resize(10,vvld(46,vld(46,0)));
        error[2].resize(10,vvld(44,vld(44,0)));
        error[3].resize(10,vvld(22,vld(22,0)));
        error[4].resize(7,vvld(10,vld(10,0))); // FC 700
        error[5].resize(1,vvld(1,vld(7,0)));
        
        f(image_no,train_size){
            //forward propogation
            forward_prop(image_no);            
            //find errors
            /*
            model : 6 layers
                    1st layer : i/p : 48*48*1
                    2nd layer : 5*5*1 conv(p=1,s=1) with 10 filters : 46*46*10
                    3rd layer : 5*5*10 conv(p=1,s=1) with 10 filters : 44*44*10
                    4th layer : max pool (/2): 22*22*10
                    5th layer : 5*5*10 conv(p=1,s=2) with 7 filters : 10*10*7 = FC 700
                    6th layer : o/p
            */
            error[5]=network[5];// output error
            assert(label[image_no]<7);
            error[5][0][0][label[image_no]]-=1.0;

            f(i,700){
                ld temp = network[4][i/100][(i/10)%10][i%10];
                f(j,7)
                    error[4][i/100][(i/10)%10][i%10]+=wFC[i][j]*error[5][0][0][i];
                error[4][i/100][(i/10)%10][i%10]*=temp*(1-temp);
            }
            
            short k,u;
            f(i,22){
                f(j,22){
                    f(k,10){
                        error[][k][][]
                    }
                }
            }
        }

        f(image_no,number_of_images){
            forward_prop(image_no);
            ld maxi=0;
            int emotion=-1;
            f(i,7)
                maxi=max(maxi,network[5][0][0][i]);
            f(i,7)
                if(maxi==network[5][0][0][i])
                    emotion=i;
            pred_label[image_no]=emotion;
        }
        f(image_no,number_of_images-train_size){
            accuracy+=(label[image_no+train_size]==pred_label[image_no+train_size]);
        }
        accuracy/=(ld)(number_of_images-train_size);
        cout<<accuracy<<'\n';
    }

}

int main(){
    ifstream train_data_file("dataset/train.csv");
    string s;
    getline(train_data_file,s);
    ll cur=0;
    // ll anger=0,disgust=0,fear=0,happy=0,sad=0,surprise=0,neutral=0,d=0;
    while(getline(train_data_file,s)){
        label[cur]=s[0]-'0';
        ll i=1,n=s.length(),j=0,kkk=48*48;
        while(kkk--){
            ll n=0,c=s[i++];
            while(c<'0'||c>'9')
                c=s[i++];        
            while(c<='9'&&c>='0'){
                n=n*10+c-'0';
                c=s[i++];
            }
            train[cur][j/48][j%48]= n;
            j++;
        }
        cur++;
    }

    // Mat image(48,48,CV_8UC1);
    // short i,j;
    // f(i,48){
    //     f(j,48){
    //         image.at<uchar>(i,j) = train[0][i][j];
    //     }
    // }
    // Size size(128,128);//the dst image size,e.g.100x100
    // resize(image,image,size);//resize image
    // namedWindow("image");
    // imshow("image",image);
    // waitKey(0);
    // input done
    
    // tested input

    back_prop();
    
}