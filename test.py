# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 18:10:32 2018

@author: lenovo
"""
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import wx
import GUI_FENLEI
import denglu
# 首先，咱们从刚刚源文件中将主窗体继承下来.就是修改过name属性的主窗体咯。
k = 0
if __name__ == '__main__':
    app = wx.App()
    # None表示的是此窗口没有上级父窗体。如果有，就直接在父窗体代码调用的时候填入‘self’就好了。
    first_win = denglu.MyFrame4(None)
    first_win.Show()
    
    app.MainLoop()
#    if(k == 1):
#        main_win = GUI_FENLEI.MyFrame3(None)
#        main_win.Show()
#        app.MainLoop()

