(Get-Content src/main.py) | ForEach-Object -Begin {=1} -Process { '{0,5}:{1}' -f ++,  }
