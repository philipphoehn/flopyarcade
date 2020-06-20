a = Analysis(['C:\\FloPyArcade\\FloPyArcadePlay.py'],
         pathex=['C:\\FloPyArcade\\test'],
         hiddenimports=[],
         hookspath=None,
         )

pyz = PYZ(a.pure)
exe = EXE(pyz,
      a.zipfiles,
      a.scripts,
      a.binaries,
      # a.datas + [('data/Sounds/Cycle.wav', 'D:\\<path>\\data\\Sounds\\Cycle.wav','DATA'),
      #  ('data/Sounds/Hold.wav', 'D:\\<path>\\data\\Sounds\\Hold.wav','DATA'),
      #  ('data/Sounds/Timer.wav', 'D:\\<path>\\data\\Sounds\\Timer.wav','DATA'),
      #  ('data/Sounds/Warn.wav', 'D:\\<path>\\data\\Sounds\\Warn.wav','DATA'),
      #  ],
      name=os.path.join('dist', 'timer.exe'),
      debug=False,
      strip=False,
      upx=False,
      # icon=r"D:\<path>\Icon.ico",
      console=True )