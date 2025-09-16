from injector import Injector
from private_gpt.settings.settings import Settings, unsafe_typed_settings

def create_application_injector() -> Injector:
    _injector = Injector(auto_bind=True)
    _injector.binder.bind(Settings, to=unsafe_typed_settings)
    return _injector
'\nGlobal injector for the application.\n\nAvoid using this reference, it will make your code harder to test.\n\nInstead, use the `request.state.injector` reference, which is bound to every request\n'
global_injector: Injector = create_application_injector()