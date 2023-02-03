from stickytape import ModuleWriterGenerator
from strip_hints import strip_string_to_string


class ModuleWriterGeneratorWithoutTyping(ModuleWriterGenerator):

    def build(self):
        output = []
        for module_path, module_source in self._modules.values():
            output.append("    __stickytape_write_module({0}, {1})\n".format(
                repr(module_path),
                repr(bytes(strip_string_to_string(str(module_source, "utf-8")), 'utf-8'))
            ))
        return "".join(output)


def _generate_module_writers(path, sys_path, add_python_modules):
    generator = ModuleWriterGeneratorWithoutTyping(sys_path)
    generator.generate_for_file(path, add_python_modules=add_python_modules)
    return generator.build()
