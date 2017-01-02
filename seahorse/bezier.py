from PIL import Image
import numpy as np


class Bezier:
    """
    Based on http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV0405/DONAVANIK/bezier.html
    """

    BEZIER = np.array([
        [-1, 3, -3, 1],
        [3, -6, 3, 0],
        [-3, 3, 0, 0],
        [1, 0, 0, 0]
    ])

    BEZIER_T = np.transpose(BEZIER)

    def generate(self, sizex, sizey):
        spacex = np.tile(np.linspace(0, 1, sizex), (sizey, 1))
        spacey = np.transpose(np.tile(np.linspace(0, 1, sizey), (sizex, 1)))
        return self.eval_bezier(self.create_geometry(), spacex, spacey)

    @staticmethod
    def eval_bezier(geometry, u, v):
        u_vec = np.array([u**3, u**2, u, 1])
        v_vec = np.array([v**3, v**2, v, 1])
        return np.dot(np.dot(np.dot(np.dot(u_vec,Bezier.BEZIER), geometry), Bezier.BEZIER_T), np.transpose(v_vec))

    @staticmethod
    def create_geometry():
        return np.random.rand(4,4)

if __name__ == '__main__':
    image = Image.merge('RGB', (
        Image.fromarray((Bezier().generate(2000,1000) * 255).astype('uint8')),
        Image.fromarray((Bezier().generate(2000,1000) * 255).astype('uint8')),
        Image.fromarray((Bezier().generate(2000,1000) * 255).astype('uint8'))))
    image.show()
