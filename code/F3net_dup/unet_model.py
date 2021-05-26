from unet_parts import *
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Unet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = U_down(64, 128)
        self.down2 = U_down(128, 256)
        self.down3 = U_down(256, 512)
        self.down4 = U_down(512, 512)
        self.up1 = U_up(1024, 256)
        self.up2 = U_up(512, 128)
        self.up3 = U_up(256, 64)
        self.up4 = U_up(128, 64)
        self.out = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x


class Res_Unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Res_Unet, self).__init__()
        self.down = RU_first_down(n_channels, 32)
        self.down1 = RU_down(32, 64)
        self.down2 = RU_down(64, 128)
        self.down3 = RU_down(128, 256)
        self.down4 = RU_down(256, 256)
        self.up1 = RU_up(512, 128)
        self.up2 = RU_up(256, 64)
        self.up3 = RU_up(128, 32)
        self.up4 = RU_up(64, 32)
        self.out = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.down(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x

class encode(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(encode, self).__init__()
        self.down = RRU_first_down(n_channels, 32)
        self.down1 = RRU_down(32, 64)
        self.down2 = RRU_down(64, 128)
        self.down3 = RRU_down(128, 256)
        self.down4 = RRU_down(256, 256)
    def forward(self, A):
        A1 = self.down(A)
        A2 = self.down1(A1)
        A3 = self.down2(A2)
        A4 = self.down3(A3)
        A5 = self.down4(A4)
        return A1,A2,A3,A4,A5

class Ringed_Res_Unet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(Ringed_Res_Unet, self).__init__()
        self.u=encode()
        self.u2=encode()
        self.up1 = RRU_up(512, 128)
        self.up2 = RRU_up(256, 64)
        self.up3 = RRU_up(128, 32)
        self.up4 = RRU_up(64, 32)
        self.out = outconv(32, n_classes)
        self.out1 = outconv(32*2, 32)
        self.out2 = outconv(64*2, 64)
        self.out3 = outconv(128*2, 128)
        self.out4 = outconv(256*2,256)
        self.out5 = outconv(256*2, 256)
        '''
        self.up12 = RRU_up(512, 128)
        self.up22 = RRU_up(256, 64)
        self.up32 = RRU_up(128, 32)
        self.up42 = RRU_up(64, 32)
        self.out2 = outconv(32, n_classes)
        '''
    def forward(self, x,x2):
        '''
        A1,A2,A3,A4,A5=self.u(x)
        A12,A22,A32,A42,A52=self.u(x2)
        x1,x2,x3,x4,x5=self.u(x)
        A1=torch.abs(A1-A12)
        A2=torch.abs(A2-A22)
        A3=torch.abs(A3-A32)
        A4=torch.abs(A4-A42)
        A5=torch.abs(A5-A52)

        A = self.up12(A5, A4)
        A = self.up22(A, A3)
        A = self.up32(A, A2)
        A = self.up42(A, A1)
        A = self.out2(A)



        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x=x*(1-torch.sigmoid(A))
        x = self.out(x)
        return x+A
        '''     
        A1,A2,A3,A4,A5=self.u(x)
        A12,A22,A32,A42,A52=self.u(x2)
        x1,x2,x3,x4,x5=self.u(x)
        A1=torch.abs(A1-A12)
        A2=torch.abs(A2-A22)
        A3=torch.abs(A3-A32)
        A4=torch.abs(A4-A42)
        A5=torch.abs(A5-A52)
        x1=self.out1(torch.cat((x1,A1),1))
        x2=self.out2(torch.cat((x2,A2),1))
        x3=self.out3(torch.cat((x3,A3),1))
        x4=self.out4(torch.cat((x4,A4),1))
        x5=self.out5(torch.cat((x5,A5),1))
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        #x=x*(1-torch.sigmoid(A))
        x = self.out(x)
        return x
