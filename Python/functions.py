def format_poly(poly):
  n = len(poly)
  str = ''
  space = ''
  for i in range(n):
    if poly[i] == 0:
      continue

    if i == 0:
      str += space + f'{poly[i]:+.5}'
    else:
      str += space + f'{poly[i]:+.5} q^-{i}'

    space = ' '

  return str