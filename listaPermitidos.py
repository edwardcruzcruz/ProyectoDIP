class Politecnicos:
    """ Lista de invitados a al labortorio CIDIS con el Mashi """

    def __init__(self):
        self.Invitados=['Aedward Cruz','Bryan Tumbaco']

    def TuSiTuNo(self,EllosSi):        
        if EllosSi in self.Invitados:
            print('Bienvenido {}'.format(EllosSi))
        else:
            print('Lo siento {}, no estas en lista'.format(EllosSi))
