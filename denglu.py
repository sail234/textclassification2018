# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 20:49:01 2018

@author: lenovo
"""

# -*- coding: utf-8 -*- 

###########################################################################
## Python code generated with wxFormBuilder (version Jun 17 2015)
## http://www.wxformbuilder.org/
##
## PLEASE DO "NOT" EDIT THIS FILE!
###########################################################################
import GUI_FENLEI
import wx
import wx.xrc
app = wx.App()
###########################################################################
## Class MyFrame4
###########################################################################

class MyFrame4 ( wx.Frame ):

    def __init__( self, parent ):
        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 500,300 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
        
        self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
        
        sbSizer2 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"label" ), wx.VERTICAL )
        
        self.m_staticText20 = wx.StaticText( sbSizer2.GetStaticBox(), wx.ID_ANY, u"用户名", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText20.Wrap( -1 )
        sbSizer2.Add( self.m_staticText20, 0, wx.ALL, 5 )
        
        self.m_textCtrl15 = wx.TextCtrl( sbSizer2.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        sbSizer2.Add( self.m_textCtrl15, 0, wx.ALL, 5 )
        
        self.m_staticText21 = wx.StaticText( sbSizer2.GetStaticBox(), wx.ID_ANY, u"密码", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText21.Wrap( -1 )
        sbSizer2.Add( self.m_staticText21, 0, wx.ALL, 5 )
        
        self.m_textCtrl16 = wx.TextCtrl( sbSizer2.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        sbSizer2.Add( self.m_textCtrl16, 0, wx.ALL, 5 )
        
        self.m_button29 = wx.Button( sbSizer2.GetStaticBox(), wx.ID_ANY, u"登录", wx.DefaultPosition, wx.DefaultSize, 0 )
        sbSizer2.Add( self.m_button29, 0, wx.ALL, 5 )
        
        
        self.SetSizer( sbSizer2 )
        self.Layout()
        
        self.Centre( wx.BOTH )
    
        # Connect Events
        self.m_button29.Bind( wx.EVT_BUTTON, self.denglu )
    
    def __del__( self ):
        pass
    
    
    # Virtual event handlers, overide them in your derived class
    def denglu( self, event ):
        username = self.m_textCtrl15.GetValue()
        password = self.m_textCtrl16.GetValue()
        print username
        print password
        if(username == "sail" and password == "123456"):
            main_win = GUI_FENLEI.MyFrame3(None)
            main_win.Show()
            app.MainLoop()
        else:
            print "username or password is wrong" 
        event.Skip()


