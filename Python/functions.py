import pysid
import pandas as pd
import control

def models_frame():
  return pd.DataFrame(columns=['model', 'na', 'nb', 'nc', 'nd', 'nf', 'nk', 'A', 'B', 'C', 'D', 'F', 'G', 'H'])

def arx(u_i, y_i, u_v, y_v, na_range, nb_range, nk_range):
  models = pd.DataFrame()
  for na in na_range:
    for nb in nb_range:
      for nk in nk_range:
        if nb-1 > na:
          continue

        id = pysid.arx(na=na, nb=nb-1, nk=nk, u=u_i, y=y_i)
        A = id.A[0][0]
        B = id.B[0][0]

        G = control.TransferFunction(B, A, dt=True)
        H = control.TransferFunction(1, A, dt=True)

        models = pd.concat([models, pd.DataFrame({
          'model': 'arx',
          'na': [na],
          'nb': [nb],
          'nk': [nk],
          'A': [A],
          'B': [B],
          'G': [G],
          'H': [H],
        })])

  return models

def armax(u_i, y_i, u_v, y_v, na_range, nb_range, nc_range, nk_range):
  models = pd.DataFrame()
  for na in na_range:
    for nb in nb_range:
      for nc in nc_range:
        for nk in nk_range:
          if nb-1 > na:
            continue

          id = pysid.armax(na=na, nb=nb-1, nc=nc, nk=nk, u=u_i, y=y_i)
          A = id.A[0][0]
          B = id.B[0][0]
          C = id.C[0]

          G = control.TransferFunction(B, A, dt=True)
          H = control.TransferFunction(C, A, dt=True)

          models = pd.concat([models, pd.DataFrame({
            'model': 'armax',
            'na': [na],
            'nb': [nb],
            'nc': [nc],
            'nk': [nk],
            'A': [A],
            'B': [B],
            'C': [C],
            'G': [G],
            'H': [H],
          })])

  return models

def oe(u_i, y_i, u_v, y_v, nb_range, nf_range, nk_range):
  models = pd.DataFrame()
  for nb in nb_range:
    for nf in nf_range:
      for nk in nk_range:
        if nb-1 > nf:
          continue

        id = pysid.oe(nb=nb-1, nf=nf, nk=nk, u=u_i, y=y_i)
        B = id.B[0][0]
        F = id.F[0][0]

        G = control.TransferFunction(B, F, dt=True)
        H = control.TransferFunction(1, 1, dt=True)

        models = pd.concat([models, pd.DataFrame({
          'model': 'oe',
          'nb': [nb],
          'nf': [nf],
          'nk': [nk],
          'B': [B],
          'F': [F],
          'G': [G],
          'H': [H],
        })])

  return models

def bj(u_i, y_i, u_v, y_v, nb_range, nc_range, nd_range, nf_range, nk_range):
  models = pd.DataFrame()
  for nb in nb_range:
    for nc in nc_range:
      for nd in nd_range:
        for nf in nf_range:
          for nk in nk_range:
            if nb-1 > nf or nc > nd:
              continue

            try:
              id = pysid.bj(nb=nb-1, nc=nc, nd=nd, nf=nf, nk=nk, u=u_i, y=y_i)
              B = id.B[0][0]
              C = id.C[0]
              D = id.D[0]
              F = id.F[0][0]

              G = control.TransferFunction(B, F, dt=True)
              H = control.TransferFunction(C, D, dt=True)

              models = pd.concat([models, pd.DataFrame({
                'model': 'bj',
                'nb': [nb],
                'nc': [nc],
                'nd': [nd],
                'nf': [nf],
                'nk': [nk],
                'B': [B],
                'C': [C],
                'D': [D],
                'F': [F],
                'G': [G],
                'H': [H],
              })])
            except Exception as e:
              # display(str(e))
              models = pd.concat([models, pd.DataFrame({
                'model': 'bj',
                'nb': [nb],
                'nc': [nc],
                'nd': [nd],
                'nf': [nf],
                'nk': [nk],
              })])

  return models
