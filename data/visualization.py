from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import py3Dmol

from data.force_field import diameter, ff_param_const


def draw3d(mol_sd):
    fig = py3Dmol.view(width=600, height=400)
    if isinstance(mol_sd, str):
        fig.addModel(mol_sd, 'sdf')
    elif isinstance(mol_sd, list):
        for m in mol_sd:
            fig.addModel(m, 'sdf')
    fig.setStyle({'stick': {'colorscheme': 'grayCarbon'}})
    return fig.render()


def vis_vdw(E_vdw, c=((0, 0, 0, 0.1),)):
    E_vdw = E_vdw.cpu()
    fig = plt.figure(1)
    plt.figure(figsize=[5, 3])
    fig.clf()
    ax = Axes3D(fig)

    radius = diameter / 2
    grid_size = E_vdw.shape[0]
    (X, Y, Z) = np.meshgrid(*[np.linspace(-radius, radius, num=grid_size) for _ in range(3)])
    ax.scatter(X, Y, Z, s=(E_vdw / ff_param_const['cut_off'])** 2, c=c)
    plt.draw()
    plt.show()


def vis_ele(E_ele,
            c_pos=((177.0/255, 34.0/255, 34.0/255, 0.1),),
            c_neg=((51.0/255, 51.0/255, 162.0/255, 0.1),)):
    E_ele = E_ele.cpu()
    plt.figure(figsize=[5, 3])
    fig = plt.figure(1)
    fig.clf()
    ax = Axes3D(fig)

    radius = diameter/2
    grid_size = E_ele.shape[0]
    (X, Y, Z) = np.meshgrid(*[np.linspace(-radius, radius, num=grid_size) for _ in range(3)])
    ax.scatter(X, Y, Z, s=np.maximum(E_ele / ff_param_const['cut_off'], 0) ** 2, c=c_pos)
    ax.scatter(X, Y, Z, s=np.maximum(-E_ele / ff_param_const['cut_off'], 0) ** 2, c=c_neg)

    plt.draw()
    plt.show()
