import './ContextMenu.css';
import {Link} from "react-router-dom";


function ContextMenu({isMenuOpened, onMenuClicked}) {

    const handleMenuClicked = event => {
        event.preventDefault()
        onMenuClicked()
    }

    return (
        <div>

            <nav className={isMenuOpened ? 'menu' : 'hideMenu'}>
                <Link to="/">Jump to Home</Link>
                <Link to="/data">View Music Dataset</Link>
            </nav>

            <div onClick={handleMenuClicked} className='contextMenu'/>
        </div>
    )
}

export default ContextMenu;