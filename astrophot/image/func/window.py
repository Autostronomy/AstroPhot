from ...backend_obj import backend


def window_or(other_origin, self_end, other_end):

    new_origin = backend.minimum(-0.5 * backend.ones_like(other_origin), other_origin)
    new_end = backend.maximum(self_end, other_end)

    return new_origin, new_end


def window_and(other_origin, self_end, other_end):
    new_origin = backend.maximum(-0.5 * backend.ones_like(other_origin), other_origin)
    new_end = backend.minimum(self_end, other_end)

    return new_origin, new_end
